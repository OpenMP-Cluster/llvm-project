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
    get_schedule_map(tdg->tdg_id).clear(); // TODO Do we really want to clear this structure?  Can it be shared between tdgs?

    if (remaining_target_tasks != 0) {
      KG_TRACE(10,("Warning: some target-tasks were discarded in favor of a new "
         "task-graph, probably due to a cancellation point.\n"));
    }

    remaining_target_tasks = 0;
  }

  // Acquire a new task graph from the target task integration layer.
  tdg = new_tdg;
  assert(tdg != nullptr);
  std::unordered_map<void *, kmp_int64> size_map;
  for (kmp_int32 idx = 0; idx < tdg->num_tasks; ++idx) {
    const kmp_node_info &task = tdg->record_map[idx];
    if (task.is_target_task) {
      // Add new target-tasks to the schedule map
      assertm(get_schedule_map(tdg->tdg_id).count(KMP_TASKNODE_TO_TASKID(task)) == 0,
              "New tasks should not be already present in the schedule map.");
      get_schedule_map(tdg->tdg_id)[KMP_TASKNODE_TO_TASKID(task)] = {TaskState::CREATED, -1, -1, -1, &task};
      remaining_target_tasks++;
      // Correct size of globally mapped variables
      if (isTargetDataTask(task)) {
        for (kmp_int32 i = 0; i < task.target_data->num_args; i++) {
          // Backup initial mapping size
          size_map[task.target_data->args[i]] = task.target_data->arg_sizes[i];
        }
      } else {
        for (kmp_int32 i = 0; i < task.target_data->num_args; i++) {
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
    KG_TRACE(10,("acquireAndScheduleGraph: RR selected\n"));
    scheduleGraphRR(scheduling_iteration_counter);
  }
  else {
    KG_TRACE(10,("acquireAndScheduleGraph: Main Scheduler selected\n"));
    scheduleGraph();
  }
  scheduling_iteration_counter++;
  isScheduleDone = 1;
  KG_TRACE(10,("acquireAndScheduleGraph Exit\n"));
}

kmp_int32 BaseScheduler::scheduleTask(kmp_int32 task_id) {	
  KG_TRACE(10,("scheduleTask start, task_id: %d\n",task_id));
  kmp_int32 active_devices_number = __kmpc_get_num_devices();
  kmp_int32 scheduled_id = -1;
  if (task_id == -1) {	  
    // If the target region is not inside a task (non-nowait), use the
    // configured scheduling strategy
    if (non_task_scheduler == NonTaskScheduler::ROUNDROBIN) {
      scheduled_id = (non_nowait_counter++ % active_devices_number);
      KG_TRACE(50,("[SCHEDULER] scheduleTask : task_id: %d, default round robin: rank = %d\n",task_id,scheduled_id));
    } else {
      // Static strategy as default: map to the first device
      scheduled_id = 0;
      KG_TRACE(50,("[SCHEDULER] scheduleTask : task_id: %d, default : rank = %d\n",task_id,scheduled_id));
    }
  } else {

    if (get_schedule_map(tdg->tdg_id).count(task_id) == 0) {
      //If the task was not scheduled (serialization), map to the first device or user-specified device
      scheduled_id = 0 ;
      KG_TRACE(50,("[SCHEDULER] scheduleTask : task_id: %d, task not scheduled : rank = %d\n",task_id,scheduled_id));
    } else { 
    	// If the target region is inside a task (nowait), use the result of the
    	// graph based scheduling algorithm.
	

	assert(get_schedule_map(tdg->tdg_id).count(task_id) == 1);

	preScheduleTask(task_id, active_devices_number);
    	const TaskScheduleData &task_schedule = get_schedule_map(tdg->tdg_id)[task_id];
    	
        KG_TRACE(50,("[SCHEDULER] scheduleTask : task_id: %d, using scheduler : rank = %d, schedule state: %d \n",task_id, task_schedule.rank, task_schedule.state));
	assert(task_schedule.rank != -1);
        assert(task_schedule.state == TaskState::SCHEDULED);
        
	scheduled_id = task_schedule.rank - 1;
        KG_TRACE(50,("[SCHEDULER] scheduleTask : task_id: %d, using scheduler : rank = %d\n",task_id,scheduled_id));
    }

  }
  assert(scheduled_id >= 0);
  assert(scheduled_id < active_devices_number);
  return scheduled_id;
}

kmp_int32 BaseScheduler::get_scheduleTask(kmp_int32 task_id){
  KG_TRACE(50,("[SCHEDULER] get_scheduleTask: task_id: %d\n",task_id));
  assert(tdg != nullptr);

  if(task_id == -1 || !get_schedule_map(tdg->tdg_id).count(task_id))
    return 0;
  return get_schedule_map(tdg->tdg_id)[task_id].rank - 1;
}

void BaseScheduler::pinTask(kmp_int32 task_id, kmp_int32 DeviceId) {
  KG_TRACE(50,("[SCHEDULER] pinTask: task_id %d; device: %d\n",task_id,DeviceId));
  if (task_id != -1) {
    TaskScheduleData &task_schedule = get_schedule_map(tdg->tdg_id)[task_id];
    task_schedule.rank = DeviceId + 1;
  }
}

void BaseScheduler::setTaskState(kmp_int32 task_id, TaskState new_state) {

  KG_TRACE(50,("[SCHEDULER] setTaskState: task_id %d;\n",task_id));
  //assert(tdg != nullptr);
  if(get_schedule_map(tdg->tdg_id).count(task_id) !=1)
 	KG_TRACE(10,("Task state fail: task %d. Initial task: %d\n",task_id,tdg->initial_task_id));
  assert(get_schedule_map(tdg->tdg_id).count(task_id) == 1);
 //assert(new_state >= TaskState::DISPATCHED);

  TaskScheduleData &task_schedule = get_schedule_map(tdg->tdg_id)[task_id];
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
  if (get_schedule_map(tdg->tdg_id).count(task_id) == 1 && new_state == TaskState::EXECUTED) {
    remaining_target_tasks--;
    KMP_DEBUG_ASSERT(remaining_target_tasks >= 0)
   }
}

void BaseScheduler::setTaskState(kmp_int32 task_id, kmp_int32 device_id,
                                 TaskState new_state) {
  // Update rank that executed, in case of failures, the translation will say
  // the real device that executed the task. Do not update if the task is FAILED
  // and the new state is EXECUTED, because this is the case where the graph was
  // just re-scheduled and we will ignore this state transition
  KG_TRACE(50,("[SCHEDULER] setTaskState: task_id %d; device: %d\n",task_id,device_id));
  if (get_schedule_map(tdg->tdg_id)[task_id].state == TaskState::FAILED &&
      new_state == TaskState::EXECUTED) {
    remaining_target_tasks--;
    return;
  }

  setTaskState(task_id, new_state);
}

// void BaseScheduler::setTaskSourceLocation(kmp_int32 task_id, ident_t *loc) {
//   TaskScheduleData &task_schedule = schedule_map[task_id];
//
//   SourceInfo tinfo(loc);
//   if (tinfo.isAvailible()) {
//     task_schedule.source_location = std::string(tinfo.getFilename()) + ":" +
//                                     std::to_string(tinfo.getLine());
//   }
// }

TaskScheduleData BaseScheduler::getTask(kmp_int32 task_id) {
  KG_TRACE(50,("[SCHEDULER] getTask: task_id %d\n",task_id));
  if (task_id != -1) {
    if (get_schedule_map(tdg->tdg_id).count(task_id) == 0) {
      return TaskScheduleData{.state = TaskState::UNDEFINED,
                              .rank = -1,
                              .translated_rank = -1,
                              .task = nullptr};
    }
    assert(get_schedule_map(tdg->tdg_id).count(task_id) == 1);

    return get_schedule_map(tdg->tdg_id)[task_id];
  }

  return TaskScheduleData{.state = TaskState::UNDEFINED,
                          .rank = -1,
                          .translated_rank = -1,
                          .task = nullptr};
}

kmp_target_dep BaseScheduler::getDep(kmp_int32 task_id, kmp_int64 addr) {
  auto dep = kmp_target_dep{.addr = -1, .dep_type = {.in = 0, .out = 0}};
  if (task_id != tdg->num_tasks) {
  kmp_node_info *task = &tdg->record_map[task_id];
    for (kmp_int32 i = 0; i < task->num_deps; ++i) {
      if (task->deps[i].addr == addr) {
        dep = task->deps[i];
        break;
      }
    }
  }

  return dep;
}

void BaseScheduler::addMemoryTasks(){

  kmp_int32 gtid = __kmp_get_gtid();
  KG_TRACE(10,("[SCHEDULER-MT] Add Memory Tasks (start) gtid: %d\n",gtid));
  tdg->num_normal_tasks.store(tdg->num_tasks.load());
  KMP_ATOMIC_ST_RLX(&__kmp_tdg_task_id, tdg->num_tasks.load());
  for (auto &map_entry : get_schedule_map(tdg->tdg_id)) {
    TaskScheduleData &task_schedule = map_entry.second;
    const kmp_node_info* task = task_schedule.task;
    if (isTargetDataTask(task))
     continue;
    kmp_int32 task_rank = task_schedule.rank;
    KG_TRACE(10,("[SCHEDULER-MT] searching successors for task %d; rank %d \n",KMP_TASKPTR_TO_TASKID(task),task_rank));
    for (kmp_int32 idx_succ = 0 ; idx_succ < task->nsuccessors ; idx_succ++) {
      kmp_node_info* successor = &(tdg->record_map[task->successors[idx_succ]]);
     if (isTargetDataTask(successor))
       continue;
    kmp_int32 successor_rank = get_schedule_map(tdg->tdg_id)[KMP_TASKPTR_TO_TASKID(successor)].rank;
    KG_TRACE(10,("[SCHEDULER-MT] successor %d; rank %d \n",KMP_TASKPTR_TO_TASKID(successor),successor_rank));
      if (successor_rank != task_rank || isTargetDataTask(task) || isTargetDataTask(successor) ) {
	if(successor->target_data != nullptr && task->target_data != nullptr) {
	  for (kmp_int32 I = 0; I < task->target_data->num_args; I++) {  
            KG_TRACE(10,("[SCHEDULER-MT] arg task: %d\n",task->target_data->args[I]));
            for (kmp_int32 J = 0; J < successor->target_data->num_args; J++) {
            KG_TRACE(10,("[SCHEDULER-MT] arg successor: %d\n",successor->target_data->args[J]));
              if (task->target_data->args_base[I] == successor->target_data->args_base[J]) {//TODO: Should we compare args or args_base?

		kmp_int32 in_rank = isTargetDataTask(task) ? omp_get_initial_device() : task_rank-1;    
		kmp_int32 out_rank = isTargetDataTask(successor) ? omp_get_initial_device() : successor_rank-1; 

          	kmp_task_t* memory_task= __kmp_create_memory_task(gtid,task->task,successor->task,in_rank, out_rank,task->target_data->args[I] ,task->target_data->arg_sizes[I],true);

		KG_TRACE(10,("[SCHEDULER-MT] Adding memory task %d ; predecessor : %d ; successor : %d \n", KMP_TASK_TO_TASKDATA(memory_task)->td_task_id,KMP_TASKPTR_TO_TASKID(task),KMP_TASKPTR_TO_TASKID(successor)));

              }
            }
          }
        }
      }
    } 
  }

  KMP_ATOMIC_ST_RLX(&__kmp_tdg_task_id, 0);

  KG_TRACE(10,("[SCHEDULER-MT] Add Memory Tasks (exit)\n"));
}


void BaseScheduler::waitScheduleGraph(){

  while(!isScheduleDone && !areTargetTaskRemaining()){
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}    

void BaseScheduler::setIsScheduleDone(bool state){
  isScheduleDone = state;
}

//Claims the scheduling for a thread
void BaseScheduler::setSchedulingThread(kmp_int32 gtid){
  if(schedulingThread == -1)
    schedulingThread = gtid;
}

void BaseScheduler::resetSchedulingThread(){
  schedulingThread = -1;
}

bool BaseScheduler::isSchedulingThread(kmp_int32 gtid){
  return schedulingThread == gtid;
}

void BaseScheduler::resetTasksState(){
  KG_TRACE(10,("resetTasksState start\n"));
  for ( auto task : get_schedule_map(tdg->tdg_id)){
    KG_TRACE(10,("resetTasksState task %d\n",task.first));
    setTaskState(task.first, TaskState::SCHEDULED ); 
    remaining_target_tasks++;
  }
  KG_TRACE(10,("resetTasksState end\n"));
}

void BaseScheduler::resetTasksRank(){
  KG_TRACE(10,("resetTasksRank start\n"));
  for ( auto& task : get_schedule_map(tdg->tdg_id)){
    task.second.rank = -1;
    task.second.translated_rank = -1;
  }
  KG_TRACE(10,("resetTasksRank end\n"));
}

void BaseScheduler::resetMemoryTasks(){
  KG_TRACE(10,("resetMemoryTasks start\n"));
  for ( kmp_int32 tdg_task_id = tdg->num_normal_tasks ; tdg_task_id < tdg->num_tasks ; tdg_task_id++ ) {
    __kmpc_remove_task(&tdg->record_map[tdg_task_id]);
  }
  tdg->num_tasks.store(tdg->num_normal_tasks.load());
  KG_TRACE(10,("resetMemoryTasks end\n"));
}

void BaseScheduler::scheduleGraphRR(kmp_int32 offset) {
  // Graph based round-robin algorithm.
  // With offset on the device
  KG_TRACE(10,("ScheduleGraphRR(Start) \n"));
  kmp_int32 task_counter = 0;

  kmp_int32 NumDevices = __kmpc_get_num_devices();
  for (auto &map_entry : get_schedule_map(tdg->tdg_id)) {
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
  for (auto &map_entry : get_schedule_map(tdg->tdg_id)) {
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    // These are already scheduled.
    if (task_schedule.state != TaskState::DEFERRED)
      continue;

    kmp_int32 Rank = 1;

    // Find an already scheduled successor
    // NOTE: Exit target data tasks are temporary scheduled but the rank will
    // be overwritten later with PinTask function.

    for (kmp_int32 idx_succ = 0 ; idx_succ < task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg->record_map[task->successors[idx_succ]]);
      auto SuccRank = get_schedule_map(tdg->tdg_id)[KMP_TASKPTR_TO_TASKID(Succ)].rank;

      if (!isClassicalTask(Succ) && SuccRank != -1) {
        Rank = SuccRank;
        break;
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    task_schedule.rank = Rank;
    task_schedule.state = TaskState::SCHEDULED;
  }

  KG_TRACE(10,("[SCHEDULER]: ScheduleGraphRR (adding memory tasks)\n"));
  addMemoryTasks();

  assert(std::none_of(get_schedule_map(tdg->tdg_id).begin(), get_schedule_map(tdg->tdg_id).end(),
                      [](const auto &entry) {
                        return entry.second.state < TaskState::SCHEDULED;
                      }));
  KG_TRACE(10,("[SCHEDULER]: ScheduleGraphRR (end) \n"));
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

  for (kmp_int32 tid = 0 ; tid < tdg->num_tasks ; tid ++ ) {
    auto task = &(tdg->record_map[tid]);
    kmp_int32 trank = get_schedule_map(tdg->tdg_id)[tid].rank;
    std::string tinfo = get_schedule_map(tdg->tdg_id)[tid].source_location;

    int task_id = KMP_TASKPTR_TO_TASKID(task);
    printf("TASKID=%d\n", task_id);

    // Print graph node 
    dot << "  V" << task_id << " [label=<<B>ID " << task_id
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << ((get_schedule_map(tdg->tdg_id)[tid].state == TaskState::SAVED) ? "<BR />Saved\""
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

    for (kmp_int32 idx = 0 ; idx < task->nsuccessors ; idx++) {
      kmp_int32 sid = task->successors[idx];
      kmp_int32 srank = get_schedule_map(tdg->tdg_id)[sid].rank;
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
  for (kmp_int32 idx = 0 ; idx<tdg.num_tasks; idx++) {
    TaskPtr Task = &(tdg.record_map[idx]);
    if (Task->npredecessors == 0)
      Set.insert(Task);
    for (kmp_int32 idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg.record_map[Task->successors[idx_succ]]);
      Incoming[Succ].push_back(Task);
    }
  }

  // Main algorithm
  while (not Set.empty()) {
    TaskPtr Task = *(Set.begin());
    Set.erase(Task);
    List.push_back(Task);
    for (kmp_int32 idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
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
