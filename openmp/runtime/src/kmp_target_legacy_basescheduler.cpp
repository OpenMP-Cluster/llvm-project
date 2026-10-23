//===-------------- basescheduler.cpp - Target-Task scheduler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the base abstract scheduler.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_legacy_basescheduler.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <unordered_set>

// Must match OMP_TGT_MAPTYPE_LITERAL from offload/include/omptarget.h.
static constexpr kmp_int64 LEGACY_MAPTYPE_LITERAL = 0x100;

LegacyBaseScheduler::LegacyBaseScheduler()
    : remaining_target_tasks(0), non_nowait_counter(0),
      non_task_scheduler(LegacyNonTaskScheduler::STATIC),
      schedulingThread(-1),graph_dump_dir_path(),
      graph_counter(0), schedule_map(), task_map(nullptr) {

  // Read environment variables
  if (char *env_str = std::getenv("OMPCLUSTER_TASK_GRAPH_DUMP_PATH")) {
    graph_dump_dir_path = env_str;
  }

  const char *str = std::getenv("OMPCLUSTER_NONTASK_SCHEDULER");
  std::string sched_str = (str != nullptr) ? str : "";

  if (sched_str == "roundrobin") {
    DP("[NON-TASK SCHEDULER] Map synchronous regions using round-robin\n");
    non_task_scheduler = LegacyNonTaskScheduler::ROUNDROBIN;
  } else {
    DP("[NON-TASK SCHEDULER] Map synchronous regions to the first worker\n");
  }
  isScheduleDone = 0;
}

LegacyBaseScheduler::~LegacyBaseScheduler() {
  // The task map can still be allocated due to a cancellation point.
  if (task_map != nullptr) {
    __kmpc_target_task_map_free(task_map);
    task_map = nullptr;

    if (remaining_target_tasks != 0) {
      DP("Warning: some target-tasks were discarted at the end of the program, "
         "probably due to a cancellation point.\n");
    }
  }
}

void LegacyBaseScheduler::setSchedulingThread(int32_t gtid) {
  if (schedulingThread == -1) {
    schedulingThread = gtid;
    isScheduleDone = 0;
  }
}

void LegacyBaseScheduler::resetSchedulingThread() {
  schedulingThread = -1;
}

bool LegacyBaseScheduler::isSchedulingThread(int32_t gtid) {
  return schedulingThread == gtid;
}

void LegacyBaseScheduler::waitOmpGraphCreation() {
  // Wait for the task builder thread (the thread executing the master/single
  // region) to finish the task graph and enter a synchronization point (starts
  // executing tasks itself).
  while (!__kmpc_is_graph_builder_tasking()) {
    // TODO: better waiting strategy. Maybe used smart sleep to free the core?
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void LegacyBaseScheduler::acquireNewGraph() {
  waitOmpGraphCreation();
  // Remove any internal state of the current task graph.
  auto temporary_task_map = __kmpc_target_task_map_create();
  if ( temporary_task_map == nullptr){
    return;
  }

  bool all_already_known = task_map != nullptr;

if (all_already_known) {
  for (int idx = 0; idx < temporary_task_map->num_tasks; ++idx) {
    const int task_id = temporary_task_map->tasks[idx].task_id;
    if (schedule_map.count(task_id) == 0) {
      all_already_known = false;
      break;
    }
  }
}



  if (all_already_known) {
  __kmpc_target_task_map_free(temporary_task_map);
  return;
}

  if (task_map != nullptr) {
    isScheduleDone = 0;
    __kmpc_target_task_map_free(task_map);
    task_map = nullptr;
    schedule_map.clear();

    if (remaining_target_tasks != 0) {
      DP("Warning: some target-tasks were discarded in favor of a new "
         "task-graph, probably due to a cancellation point.\n");
    }
    remaining_target_tasks = 0;
  }

  // Wait for the new graph to be created by the OpenMP runtime.
  waitOmpGraphCreation();

  // Acquire a new task graph from the target task integration layer.
  task_map = temporary_task_map;
  assert(task_map != nullptr);

  std::unordered_map<void *, int64_t> size_map;

// Pass 1: collect all sizes provided by target-data tasks.
for (int idx = 0; idx < task_map->num_tasks; ++idx) {
  const kmp_target_task_t &task = task_map->tasks[idx];

  if (!task.is_target_task)
    continue;

  if (!isLegacyTargetDataTask(task))
    continue;

  for (int32_t i = 0; i < task.target_data.num_args; ++i)
    size_map[task.target_data.args[i]] = task.target_data.arg_sizes[i];
}

// Pass 2: initialize scheduler state and propagate sizes to compute tasks.
for (int idx = 0; idx < task_map->num_tasks; ++idx) {
  const kmp_target_task_t &task = task_map->tasks[idx];

  if (!task.is_target_task)
    continue;

  assertm(schedule_map.count(task.task_id) == 0,
          "Task already present in legacy schedule map");

  schedule_map[task.task_id] = {
      LegacyTaskState::CREATED, -1, -1, -1, &task};

  remaining_target_tasks++;

  if (!isLegacyTargetDataTask(task)) {
    for (int32_t i = 0; i < task.target_data.num_args; ++i) {
      auto it = size_map.find(task.target_data.args[i]);
      if (it != size_map.end())
        task.target_data.arg_sizes[i] = it->second;
    }
  }
}
  if (graph_counter != 0) {
    graph_counter++;
  }
  dumpTaskGraph();
  graph_counter++;
  acquireNewGraphPost();
}

void LegacyBaseScheduler::acquireAndScheduleGraph() {
  // Acquire a new graph and reset the scheduler when we have no task scheduled.
  // Do not try to acquire a new graph when there are no active tasks, as this
  // would lead to a deadlock.
  if (!areTargetTaskRemaining() && __kmpc_has_active_tasks()) {
    acquireNewGraph();
  } else {
    if (!__kmpc_has_active_tasks())
      resetSchedulingThread();
    return;
  }

  if (task_map == nullptr)
    return;

  int task_id = task_map->tasks[0].task_id;
  const LegacyTaskScheduleData &task_schedule = schedule_map[task_id];

  // Schedule the graph if the current task is not scheduled yet.
  if (task_schedule.state < LegacyTaskState::SCHEDULED) {
    scheduleGraph();

    for (const auto &task : Tasks(task_map)) {
      const int src_rank = schedule_map.at(task.task_id).rank;

      for (const auto &succ : Successors(task)) {
        const int dst_rank = schedule_map.at(succ.task_id).rank;

        if (src_rank == dst_rank)
          continue;

        // Same policy as the current TDG addMemoryTasks():
        // do not insert memory tasks before target-data successors.
        if (succ.target_data.outlined_fn_id == nullptr)
          continue;

        for (int32_t i = 0; i < task.target_data.num_args; ++i) {
          for (int32_t j = 0; j < succ.target_data.num_args; ++j) {
            if (task.target_data.args_base[i] !=
                succ.target_data.args_base[j])
              continue;

            kmp_taskdata_t *src_td = KMP_TASK_TO_TASKDATA(task.task);
            kmp_taskdata_t *dst_td = KMP_TASK_TO_TASKDATA(succ.task);

            if (task.target_data.args_base[i] == nullptr ||
                task.target_data.args[i] == nullptr ||
                (task.target_data.arg_types[i] & LEGACY_MAPTYPE_LITERAL))
              continue;

            bool carries_dependency = false;

            for (int32_t sd = 0;
                 sd < task.num_deps && !carries_dependency; ++sd) {
              if (!task.deps[sd].dep_type.out)
                continue;

              if ((void *)task.deps[sd].addr != task.target_data.args_base[i])
                continue;

              for (int32_t dd = 0; dd < succ.num_deps; ++dd) {
                if (!succ.deps[dd].dep_type.in)
                  continue;

                if (task.deps[sd].addr == succ.deps[dd].addr) {
                  carries_dependency = true;
                  break;
                }
              }
            }

            if (!carries_dependency)
              continue;

            bool dep_edge_found = false;

            if (src_td->td_depnode && dst_td->td_depnode) {
              for (kmp_depnode_list_t *p = src_td->td_depnode->dn.successors;
                   p != nullptr; p = p->next) {
                if (p->node == dst_td->td_depnode) {
                  dep_edge_found = true;
                  break;
                }
              }
            }

            if (!dep_edge_found)
              continue;

            kmp_task_t *memory_task = __kmp_insert_legacy_memory_task(
                __kmp_get_gtid(), task.task, succ.task, src_rank - 1,
                dst_rank - 1, task.target_data.args[i],
                task.target_data.arg_sizes[i]);

            if (memory_task)
              break;
          }
        }
      }
    }

    isScheduleDone = 1;
  }
}

int32_t LegacyBaseScheduler::scheduleTask(int32_t task_id) {
  int active_devices_number = omp_get_num_devices();

  preScheduleTask(task_id, active_devices_number);
  int scheduled_id = -1;
  if (task_id == -1) {
    // If the target region is not inside a task (non-nowait), use the
    // configured scheduling strategy
    if (non_task_scheduler == LegacyNonTaskScheduler::ROUNDROBIN) {
      scheduled_id = (non_nowait_counter++ % active_devices_number);
    } else {
      // Static strategy as default: map to the first device
      scheduled_id = 0;
    }
  } else {
    // If the target region is inside a task (nowait), use the result of the
    // graph based scheduling algorithm.
    assert(schedule_map.count(task_id) == 1);
    const LegacyTaskScheduleData &task_schedule = schedule_map[task_id];

    assert(task_schedule.rank != -1);
    assert(task_schedule.state == LegacyTaskState::SCHEDULED ||
           task_schedule.state == LegacyTaskState::FAILED);

    scheduled_id = task_schedule.rank - 1;
  }

  assert(scheduled_id >= 0);
  assert(scheduled_id < active_devices_number);

  return scheduled_id;
}

void LegacyBaseScheduler::pinTask(int32_t task_id, int LegacyDeviceId) {
  if (task_id != -1) {
    LegacyTaskScheduleData &task_schedule = schedule_map[task_id];
    task_schedule.rank = LegacyDeviceId + 1;
  }
}

void LegacyBaseScheduler::setTaskState(int32_t task_id, LegacyTaskState new_state) {
  assert(task_map != nullptr);
  assert(schedule_map.count(task_id) == 1);
  assert(new_state >= LegacyTaskState::DISPATCHED);

  LegacyTaskScheduleData &task_schedule = schedule_map[task_id];

  // If the current task state is FAILED, it means that the counter was already
  // decremented for this task, so we increment again to make the values
  // consistent
  if (task_schedule.state == LegacyTaskState::FAILED) {
    // A FAILED task cannot have its state changed directly to EXECUTED; it
    // needs to be DISPATCHED first.
    remaining_target_tasks++;
  }

  task_schedule.state = new_state;
  if (new_state == LegacyTaskState::EXECUTED) {
    remaining_target_tasks--;
    assert(remaining_target_tasks >= 0);
  }
}

void LegacyBaseScheduler::setTaskState(int32_t task_id, int device_id,
                                 LegacyTaskState new_state) {
  // Update rank that executed, in case of failures, the translation will say
  // the real device that executed the task. Do not update if the task is FAILED
  // and the new state is EXECUTED, because this is the case where the graph was
  // just re-scheduled and we will ignore this state transition
  if (schedule_map[task_id].state == LegacyTaskState::FAILED &&
      new_state == LegacyTaskState::EXECUTED) {
    remaining_target_tasks--;
    return;
  }

  schedule_map[task_id].rank = device_id + 1;
  setTaskState(task_id, new_state);
}

LegacyTaskScheduleData LegacyBaseScheduler::getTask(int task_id) {
  if (task_id != -1) {
    if (schedule_map.count(task_id) == 0) {
      acquireNewGraph();
    }
    assert(schedule_map.count(task_id) == 1);

    return schedule_map[task_id];
  }

  return LegacyTaskScheduleData{.state = LegacyTaskState::UNDEFINED,
                          .rank = -1,
                          .translated_rank = -1,
                          .task = nullptr};
}

bool LegacyBaseScheduler::isTargetDataTask(int32_t task_id) {
  kmp_target_task_t *task =
      __kmpc_target_task_map_get(task_map, task_id);

  if (task == nullptr)
    return false;

  return isLegacyTargetDataTask(*task);
}

kmp_target_dep LegacyBaseScheduler::getDep(int32_t task_id, int64_t addr) {
  auto dep = kmp_target_dep{.addr = -1, .dep_type = {.in = 0, .out = 0}};

  kmp_target_task_t *task = __kmpc_target_task_map_get(task_map, task_id);

  if (task != nullptr) {
    for (int i = 0; i < task->num_deps; ++i) {
      if (task->deps[i].addr == addr) {
        dep = task->deps[i];
        break;
      }
    }
  }

  return dep;
}

bool LegacyBaseScheduler::waitScheduleGraph() {
  while (!isScheduleDone && __kmpc_has_active_tasks() &&
         !areTargetTaskRemaining()) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  return isScheduleDone;
}

void LegacyBaseScheduler::setIsScheduleDone(bool state) {
  isScheduleDone = state;
}

void LegacyBaseScheduler::dumpTaskGraph() {
  if (graph_dump_dir_path.empty())
    return;

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  graph_counter++;
  // Print header
  dot << "// Generated in LegacyBaseScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"Generated in LegacyBaseScheduler\";\n";

  if (task_map == nullptr) {
    dot << "}\n";
    return;
  }

  for (const auto &task : Tasks(task_map)) {
    int32_t tid = task.task_id;
    int32_t trank = schedule_map[tid].rank;
    std::string tinfo = schedule_map[tid].source_location;

    // Print graph node
    dot << "  V" << tid << " [label=<<B>ID " << tid
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << ((schedule_map[tid].state == LegacyTaskState::SAVED) ? "<BR />Saved\""
                                                          : "<BR />Unsaved\"");

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    dot << "</FONT>>";

    if (isLegacyTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isLegacyRootTask(task) || isLegacyExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    for (const auto &succ : Successors(task)) {
      int32_t sid = succ.task_id;
      int32_t srank = schedule_map[sid].rank;
      const auto *color = detail::getLegacyEdgeColor(trank, srank);

      // Print graph edge
      dot << "  V" << tid << " -> V" << sid << " [color=\"" << color
          << "\"];\n";
    }
  }

  dot << "}\n";
}

// Task Iterators
// =============================================================================

std::list<LegacyTaskPtr> TopologicalOrder(kmp_target_task_map_t &TaskMap) {
  // Set of all nodes with no incoming edge
  std::unordered_set<LegacyTaskPtr> Set;
  // Hash map to keep track of incoming edges
  std::unordered_map<LegacyTaskPtr, std::list<LegacyTaskPtr>> Incoming;
  // Empty list that will contain the sorted elements
  std::list<LegacyTaskPtr> List;

  Set.reserve(TaskMap.num_tasks);
  Incoming.reserve(TaskMap.num_tasks);

  // Initialization
  for (const auto &Task : Tasks(TaskMap)) {
    if (Task.predecessors == nullptr)
      Set.insert(&Task);
    for (const auto &Succ : Successors(Task))
      Incoming[&Succ].push_back(&Task);
  }

  // Main algorithm
  while (not Set.empty()) {
    LegacyTaskPtr Task = *(Set.begin());
    Set.erase(Task);
    List.push_back(Task);
    for (const auto &Succ : Successors(Task)) {
      Incoming[&Succ].remove(Task);
      if (Incoming[&Succ].empty())
        Set.insert(&Succ);
    }
  }

  assert(std::all_of(Incoming.begin(), Incoming.end(),
                     [](const auto &pair) { return pair.second.empty(); }) &&
         "There should be no cyclic dependencies in the task graph");

  assert(std::all_of(List.begin(), List.end(),
                     [](LegacyTaskPtr item) { return item != nullptr; }) &&
         "There should be no null pointers in the topologically ordered list");

  return List;
}

std::list<LegacyTaskPtr> TopologicalOrder(kmp_target_task_map_t *TaskMap) {
  assert(TaskMap != nullptr);
  return TopologicalOrder(*TaskMap);
}

std::list<LegacyTaskPtr> ReverseTopologicalOrder(kmp_target_task_map_t &TaskMap) {
  auto List = TopologicalOrder(TaskMap);
  List.reverse();
  return List;
}

std::list<LegacyTaskPtr> ReverseTopologicalOrder(kmp_target_task_map_t *TaskMap) {
  assert(TaskMap != nullptr);
  return ReverseTopologicalOrder(*TaskMap);
}
