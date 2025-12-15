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

#include "kmp_target_basescheduler.h"
#include "kmp.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <stdio.h>

BaseScheduler::BaseScheduler()
    : remaining_target_tasks(0), non_nowait_counter(0),
      non_task_scheduler(NonTaskScheduler::STATIC), graph_dump_dir_path(),
      graph_counter(0) {

  // Read environment variables
  if (char *env_str = std::getenv("OMPCLUSTER_TASK_GRAPH_DUMP_PATH")) {
    graph_dump_dir_path = env_str;
  }

  const char *str = std::getenv("OMPCLUSTER_NONTASK_SCHEDULER");
  std::string sched_str = (str != nullptr) ? str : "";

  if (sched_str == "roundrobin") {
    KG_TRACE(10,("[NON-TASK SCHEDULER] Map synchronous regions using round-robin\n"));
    non_task_scheduler = NonTaskScheduler::ROUNDROBIN;
  } else {
    KG_TRACE(10,("[NON-TASK SCHEDULER] Map synchronous regions to the first worker\n"));
  }
  
  //if policy = 0 (default), use selected scheduler for initial iterations; 
  //if policy = 1, use RR with offset (equal to iteration's number)
  if(char *env_str = std::getenv("OMPCLUSTER_INITIAL_SCHEDULE_POLICY")) {
    initial_schedule_policy = std::stoi(env_str);
  } 
  else {
    initial_schedule_policy = 0;
  }
  
  //Set up the number of initial iterations (after these initial iterations, we schedule at least one) 
  if(char *env_str = std::getenv("OMPCLUSTER_INITIAL_ITERATION_NUMBER")) {
    initial_iteration = std::stoi(env_str);
  } 
  else {
    initial_iteration = 0;
  }
   
  //Reschedule every X iterations
  // if 0 (default) do not reschedule
  if(char *env_str = std::getenv("OMPCLUSTER_RESCHEDULING_INTERVAL")) {
    rescheduling_interval = std::stoi(env_str);
  } 
  else {
    rescheduling_interval = 0;
  }
  scheduling_iteration_counter = 0;
  isScheduleDone = 0;
}

BaseScheduler::~BaseScheduler() {
  // The task map can still be allocated due to a cancellation point.
  if (tdg != nullptr) {
    if (remaining_target_tasks != 0) {
      KG_TRACE(10,("Warning: some target-tasks were discarted at the end of the program, "
         "probably due to a cancellation point.\n"));
    }
  }
}

void BaseScheduler::waitOmpGraphCreation() {
  // Wait for the task builder thread (the thread executing the master/single
  // region) to finish the task graph and enter a synchronization point (starts
  // executing tasks itself).
  while (!__kmpc_is_graph_builder_tasking() && !isScheduleDone &&  __kmpc_has_active_tasks()) {
    // TODO: better waiting strategy. Maybe used smart sleep to free the core?
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void BaseScheduler::acquireNewGraph(kmp_tdg_info_t* new_tdg) {

  KG_TRACE(10,("acquireNewGraph Start\n"));

  // Remove any internal state of the current task graph.
  
  if (tdg != nullptr) {
   //Returning if we're still in the same tdg
    if (tdg == new_tdg){
      KG_TRACE(10,("acquireNewGraph Exit : Same tdg\n"));
      return;
    }
    isScheduleDone = 0;
    tdg->schedule_map.clear(); // TODO Do we really want to clear this structure?  Can it be shared between tdgs?

    if (remaining_target_tasks != 0) {
      KG_TRACE(10,("Warning: some target-tasks were discarded in favor of a new "
         "task-graph, probably due to a cancellation point.\n"));
    }

    remaining_target_tasks = 0;
  }

  // Acquire a new task graph from the target task integration layer.
  tdg = new_tdg;
  assert(tdg != nullptr);
  std::unordered_map<void *, int64_t> size_map;
  for (int idx = 0; idx < tdg->num_tasks; ++idx) {
    const kmp_node_info &task = tdg->record_map[idx];
    if (task.is_target_task) {
      // Add new target-tasks to the schedule map
      assertm(tdg->schedule_map.count(KMP_TASKNODE_TO_TASKID(task)) == 0,
              "New tasks should not be already present in the schedule map.");
      tdg->schedule_map[KMP_TASKNODE_TO_TASKID(task)] = {TaskState::CREATED, -1, -1, -1, &task};
      remaining_target_tasks++;
      // Correct size of globally mapped variables
      if (isTargetDataTask(task)) {
        for (int32_t i = 0; i < task.target_data->num_args; i++) {
          // Backup initial mapping size
          size_map[task.target_data->args[i]] = task.target_data->arg_sizes[i];
        }
      } else {
        for (int32_t i = 0; i < task.target_data->num_args; i++) {
          // Update mapping size when needed
          auto it = size_map.find(task.target_data->args[i]);
          if (it != size_map.end())
            task.target_data->arg_sizes[i] = it->second;
        }
      }
    }
  }
  if (graph_counter != 0) {
    graph_counter++;
  }
  dumpTaskGraph();
  graph_counter++;
  KG_TRACE(10,("acquireNewGraph Exit\n"));
}

void BaseScheduler::acquireAndScheduleGraph(kmp_tdg_info_t* tdg) {
  // Acquire a new graph and reset the scheduler when we have no task scheduled.

  KG_TRACE(10,("acquireAndScheduleGraph Start\n"));
  acquireNewGraph(tdg);
  assertm(tdg != nullptr, "Task Map should not be null at this point");

  // Schedule the graph.
  if(initial_schedule_policy == 1 && scheduling_iteration_counter < initial_iteration) {
    scheduleGraphRR(scheduling_iteration_counter);
  }
  else {
    scheduleGraph();
  }
  scheduling_iteration_counter++;
  isScheduleDone = 1;
  KG_TRACE(10,("acquireAndScheduleGraph Exit\n"));
}

int32_t BaseScheduler::scheduleTask(int32_t task_id) {

  KG_TRACE(10,("scheduleTask start, task_id: %d\n",task_id));
  int active_devices_number = __kmpc_get_num_devices();
  int scheduled_id = -1;
  if (task_id == -1) {	  
    // If the target region is not inside a task (non-nowait), use the
    // configured scheduling strategy
    if (non_task_scheduler == NonTaskScheduler::ROUNDROBIN) {
      scheduled_id = (non_nowait_counter++ % active_devices_number);
    } else {
      // Static strategy as default: map to the first device
      scheduled_id = 0;
    }
  } else {

    if (tdg->schedule_map.count(task_id) == 0) {
      //If the task was not scheduled (serialization), map to the first device or user-specified device
      scheduled_id = 0 ;
    } else { 
    	// If the target region is inside a task (nowait), use the result of the
    	// graph based scheduling algorithm.
	
	assert(tdg->schedule_map.count(task_id) == 1);
        
	preScheduleTask(task_id, active_devices_number);
    	const TaskScheduleData &task_schedule = tdg->schedule_map[task_id];
    	
	assert(task_schedule.rank != -1);
        assert(task_schedule.state == TaskState::SCHEDULED);
        
	scheduled_id = task_schedule.rank - 1;
    }

  }
  assert(scheduled_id >= 0);
  assert(scheduled_id < active_devices_number);
  return scheduled_id;
}

kmp_int32 BaseScheduler::get_scheduleTask(kmp_int32 task_id){
  if(task_id == -1 || !tdg->schedule_map.count(task_id))
    return 0;
  return tdg->schedule_map[task_id].rank - 1;
}

void BaseScheduler::pinTask(int32_t task_id, int DeviceId) {
  if (task_id != -1) {
    TaskScheduleData &task_schedule = tdg->schedule_map[task_id];
    task_schedule.rank = DeviceId + 1;
  }
}

void BaseScheduler::setTaskState(int32_t task_id, TaskState new_state) {

  //assert(tdg != nullptr);
  if(tdg->schedule_map.count(task_id) !=1)
 	KG_TRACE(10,("Task state fail: task %d. Initial task: %d\n",task_id,tdg->initial_task_id));
  assert(tdg->schedule_map.count(task_id) == 1);
 //assert(new_state >= TaskState::DISPATCHED);

  TaskScheduleData &task_schedule = tdg->schedule_map[task_id];
  // If the current task state is FAILED, it means that the counter was already
  // decremented for this task, so we increment again to make the values
  // consistent
  if (task_schedule.state == TaskState::FAILED) {
    // A FAILED task cannot have its state changed directly to EXECUTED, it
    // needs to be DISPATCHED first
    // if (new_state != TaskState::DISPATCHED)
    remaining_target_tasks++;
  }

  task_schedule.state = new_state;
  if (tdg->schedule_map.count(task_id) == 1 && new_state == TaskState::EXECUTED) {
    remaining_target_tasks--;
    KMP_DEBUG_ASSERT(remaining_target_tasks >= 0)
   }
}

void BaseScheduler::setTaskState(int32_t task_id, int device_id,
                                 TaskState new_state) {
  // Update rank that executed, in case of failures, the translation will say
  // the real device that executed the task. Do not update if the task is FAILED
  // and the new state is EXECUTED, because this is the case where the graph was
  // just re-scheduled and we will ignore this state transition
  if (tdg->schedule_map[task_id].state == TaskState::FAILED &&
      new_state == TaskState::EXECUTED) {
    remaining_target_tasks--;
    return;
  }

  tdg->schedule_map[task_id].rank = device_id + 1;
  setTaskState(task_id, new_state);
}

// void BaseScheduler::setTaskSourceLocation(int32_t task_id, ident_t *loc) {
//   assert(task_map != nullptr);
//
//   TaskScheduleData &task_schedule = schedule_map[task_id];
//
//   SourceInfo tinfo(loc);
//   if (tinfo.isAvailible()) {
//     task_schedule.source_location = std::string(tinfo.getFilename()) + ":" +
//                                     std::to_string(tinfo.getLine());
//   }
// }

TaskScheduleData BaseScheduler::getTask(int task_id) {
  if (task_id != -1) {
    if (tdg->schedule_map.count(task_id) == 0) {
      //acquireNewGraph(); TODO
    }
    if (tdg->schedule_map.count(task_id) == 0) {
      return TaskScheduleData{.state = TaskState::UNDEFINED,
                              .rank = -1,
                              .translated_rank = -1,
                              .task = nullptr};
    }
    assert(tdg->schedule_map.count(task_id) == 1);

    return tdg->schedule_map[task_id];
  }

  return TaskScheduleData{.state = TaskState::UNDEFINED,
                          .rank = -1,
                          .translated_rank = -1,
                          .task = nullptr};
}

kmp_target_dep BaseScheduler::getDep(int32_t task_id, int64_t addr) {
  auto dep = kmp_target_dep{.addr = -1, .dep_type = {.in = 0, .out = 0}};
  if (task_id != tdg->num_tasks) {
  kmp_node_info *task = &tdg->record_map[task_id];
    for (int i = 0; i < task->num_deps; ++i) {
      if (task->deps[i].addr == addr) {
        dep = task->deps[i];
        break;
      }
    }
  }

  return dep;
}

void BaseScheduler::addMemoryTasks(){
  KG_TRACE(10,("[SCHEDULER] Add Memory Tasks (begin)\n"));
  printf("[SCHEDULER] ScheduleMap size: %d\n", tdg->schedule_map.size());
  for (auto &map_entry : tdg->schedule_map) {
    TaskScheduleData &task_schedule = map_entry.second;
    kmp_node_info* task = task_schedule.task;
    kmp_int32 task_rank = task_schedule.rank;

    printf("[SCHEDULER] Trying to add memory task for task %d on device %d\n",map_entry.first,task_schedule.rank);
    if (isTargetDataTask(task))
      continue;
    
    printf("[SCHEDULER] Trying to add memory task for task %d on device %d\n",map_entry.first,task_schedule.rank);
    for (kmp_int32 idx_succ = 0 ; idx_succ < task->nsuccessors ; idx_succ++) {
      kmp_node_info* successor = &(tdg->record_map[task->successors[idx_succ]]);

      if (isTargetDataTask(successor))
        continue;

      kmp_int32 successor_rank = tdg->schedule_map[KMP_TASKPTR_TO_TASKID(successor)].rank;
      printf("[SCHEDULER] Found successor %d on device %d\n",KMP_TASKPTR_TO_TASKID(successor),successor_rank);
      
      if (successor_rank != task_rank) {
	if(successor->target_data !=nullptr && task->target_data!=nullptr) {
      
	  for (kmp_int32 I = 0; I < task->target_data->num_args; I++) {
            for (kmp_int32 J = 0; J < successor->target_data->num_args; J++) {
              if (successor->target_data->args[I] == successor->target_data->args[J]) {
                printf("[SCHEDULER] Adding memory task between task %d and task %d ",KMP_TASKPTR_TO_TASKID(task),KMP_TASKPTR_TO_TASKID(successor));
          	kmp_task_t* memory_task= __kmp_create_memory_task(0,task->task,successor->task,task_rank, successor_rank,task->target_data->args[I] ,task->target_data->arg_sizes[I],true);
        	kmp_task_and_pred temp;
		temp.pred = task->task;
		temp.task = memory_task;
		tdg->memory_task_vec.push_back(temp);
              }
            }
          }
        }
      }
    } 
  }

  KG_TRACE(10,("[SCHEDULER] Add Memory Tasks (exit)\n"));
}


void BaseScheduler::waitScheduleGraph(){

  while(!isScheduleDone && __kmpc_has_active_tasks() && !areTargetTaskRemaining()){
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}    

void BaseScheduler::setIsScheduleDone(bool state){
  isScheduleDone = state;
}

//Claims the scheduling for a thread
void BaseScheduler::setSchedulingThread(int32_t gtid){
  if(schedulingThread == -1)
    schedulingThread = gtid;
}

void BaseScheduler::resetSchedulingThread(){
  schedulingThread = -1;
}

bool BaseScheduler::isSchedulingThread(int32_t gtid){
  return schedulingThread == gtid;
}

void BaseScheduler::resetTasksState(){
  KG_TRACE(10,("resetTasksState start\n"));
  for ( auto task : tdg->schedule_map){
    KG_TRACE(10,("resetTasksState task %d\n",task.first));
    setTaskState(task.first, TaskState::SCHEDULED ); 
    remaining_target_tasks++;
  }
  KG_TRACE(10,("resetTasksState end\n"));
}

void BaseScheduler::resetTasksRank(){
  for ( auto& task : tdg->schedule_map){
    task.second.rank = -1;
    task.second.translated_rank = -1;
  }
}

void BaseScheduler::resetMemoryTasks(){
  for (auto& [pred,task] : tdg->memory_task_vec){
    __kmpc_remove_task(task,pred);
  }
  tdg->memory_task_vec.clear();
}

void BaseScheduler::scheduleGraphRR(kmp_int32 offset) {
  // Graph based round-robin algorithm.
  // With offset on the device
  int task_counter = 0;

  int NumDevices = __kmpc_get_num_devices();
  for (auto &map_entry : tdg->schedule_map) {
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    assert(task_schedule.task->is_target_task);
    assert(task_schedule.state <= TaskState::SCHEDULED);

    if (isTargetDataTask(task)) {
      // Defer the scheduling for target data tasks
      task_schedule.state = TaskState::DEFERRED;
    }
    else {
      task_schedule.state = TaskState::SCHEDULED;
      task_schedule.rank = ((task_counter++ + offset) % NumDevices) + 1;
    }
 }

  // Schedule deferred tasks
  for (auto &map_entry : tdg->schedule_map) {
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    // These are already scheduled.
    if (task_schedule.state != TaskState::DEFERRED)
      continue;

    int32_t Rank = 1;

    // Find an already scheduled successor
    // NOTE: Exit target data tasks are temporary scheduled but the rank will
    // be overwritten later with PinTask function.

    for (int32_t idx_succ = 0 ; idx_succ < task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg->record_map[task->successors[idx_succ]]);
      auto SuccRank = tdg->schedule_map[KMP_TASKPTR_TO_TASKID(Succ)].rank;

      if (!isClassicalTask(Succ) && SuccRank != -1) {
        Rank = SuccRank;
        break;
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    task_schedule.rank = Rank;
    task_schedule.state = TaskState::SCHEDULED;
  }

  addMemoryTasks();

  assert(std::none_of(tdg->schedule_map.begin(), tdg->schedule_map.end(),
                      [](const auto &entry) {
                        return entry.second.state < TaskState::SCHEDULED;
                      }));
}

bool BaseScheduler::is_rescheduling(){
  if(scheduling_iteration_counter <= initial_iteration || (rescheduling_interval && (scheduling_iteration_counter - initial_iteration) % rescheduling_interval == 0))
    return true;
  return false;
}
  
void BaseScheduler::dumpTaskGraph() {
  if (graph_dump_dir_path.empty())
    return;

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  graph_counter++;
  // Print header
  dot << "// Generated in BaseScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"Generated in BaseScheduler\";\n";

  if (tdg == nullptr) {
    dot << "}\n";
    return;
  }

  for (int32_t tid = 0 ; tid < tdg->num_tasks ; tid ++ ) {
    auto task = &(tdg->record_map[tid]);
    int32_t trank = tdg->schedule_map[tid].rank;
    std::string tinfo = tdg->schedule_map[tid].source_location;

    // Print graph node 
    dot << "  V" << tid << " [label=<<B>ID " << tid
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << ((tdg->schedule_map[tid].state == TaskState::SAVED) ? "<BR />Saved\""
                                                          : "<BR />Unsaved\"");

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    dot << "</FONT>>";

    if (isTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isRootTask(task) || isExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    for (int32_t idx = 0 ; idx < task->nsuccessors ; idx++) {
      int32_t sid = task->successors[idx];
      int32_t srank = tdg->schedule_map[sid].rank;
      const auto *color = detail::getEdgeColor(trank, srank);
      // Print graph edge
      dot << "  V" << tid << " -> V" << sid << " [color=\"" << color
          << "\"];\n";
    }
  }

  dot << "}\n";
}

// Task Iterators
// =============================================================================

std::list<TaskPtr> TopologicalOrder(kmp_tdg_info &tdg) {
  // Set of all nodes with no incoming edge
  std::unordered_set<TaskPtr> Set;
  // Hash map to keep track of incoming edges
  std::unordered_map<TaskPtr, std::list<TaskPtr>> Incoming;
  // Empty list that will contain the sorted elements
  std::list<TaskPtr> List;

  Set.reserve(tdg.num_tasks);
  Incoming.reserve(tdg.num_tasks);

  // Initialization
  for (int32_t idx = 0 ; idx<tdg.num_tasks; idx++) {
    TaskPtr Task = &(tdg.record_map[idx]);
    if (Task->npredecessors == 0)
      Set.insert(Task);
    for (int32_t idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg.record_map[Task->successors[idx_succ]]);
      Incoming[Succ].push_back(Task);
    }
  }

  // Main algorithm
  while (not Set.empty()) {
    TaskPtr Task = *(Set.begin());
    Set.erase(Task);
    List.push_back(Task);
    for (int32_t idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg.record_map[Task->successors[idx_succ]]);
      Incoming[Succ].remove(Task);
      if (Incoming[Succ].empty())
        Set.insert(Succ);
    }
  }

  assert(std::all_of(Incoming.begin(), Incoming.end(),
                     [](const auto &pair) { return pair.second.empty(); }) &&
         "There should be no cyclic dependencies in the task graph");

  assert(std::all_of(List.begin(), List.end(),
                     [](TaskPtr item) { return item != nullptr; }) &&
         "There should be no null pointers in the topologically ordered list");

  return List;
}

std::list<TaskPtr> TopologicalOrder(kmp_tdg_info *tdg) {
  assert(tdg != nullptr);
  return TopologicalOrder(*tdg);
}

std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info &tdg) {
  auto List = TopologicalOrder(tdg);
  List.reverse();
  return List;
}

std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info *tdg) {
  assert(tdg != nullptr);
  return ReverseTopologicalOrder(*tdg);
}
