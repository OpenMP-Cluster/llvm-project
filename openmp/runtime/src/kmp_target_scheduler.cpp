//===----------------- kmp_target_scheduler.cpp - Target-Task scheduler
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduler used to manage multiple devices of the same type.
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_target_basescheduler.h"
#include "kmp_target_heft.h"
#include "kmp_target_graph_roundrobin.h"
#include "kmp_target_roundrobin.h"

#if ENABLE_LIBOMPTARGET
static int (*omp_get_num_devices)(void) = nullptr;
static bool init = 0;

void __kmp_init_omp_get_num_devices(){
   *(void **)(&omp_get_num_devices) = KMP_DLSYM("omp_get_num_devices");
}
#endif

int __kmpc_get_num_devices(){
  if (!init){
    init = 1;
    __kmp_init_omp_get_num_devices();
  }
  return omp_get_num_devices();
}

// Strings allowed to be passed in the OMPCLUSTER_SCHEDULER environment
// variable.
// NOTE: Update this when adding a new scheduler.
static const char *allowed_schedulers[] = {
    "roundrobin",
    "graph_roundrobin",
    "heft",
};

// Scheduler global state
static BaseScheduler *scheduler;

// The default scheduler used by the MPI plugin
using DefaultScheduler = HEFTScheduler;

// Scheduler implementation
// =============================================================================

static inline void WarnUnknownScheduler(const std::string &str) {
  std::string message;
  for (const auto &name : allowed_schedulers) {
    message += " ";
    message += name;
  }

  KG_TRACE(10,("[SCHEDULER] Invalid scheduler OMPCLUSTER_SCHEDULER=%s\n", str.c_str()));
  KG_TRACE(10,("[SCHEDULER] Allowed values are:%s\n", message.c_str()));
}

/// Initialize the scheduler.
///
/// \note callers are encoraged to check the scheduler pointer before calling
/// this function, thus avoiding the need to acquire a lock.
void __kmpc_InitScheduler() {
  // Only one thread can try to init the scheduler at a time.
  static std::mutex init_mtx;
  std::unique_lock<std::mutex> init_lk(init_mtx);

  if (scheduler != nullptr) {
    return;
  }

  const char *str = std::getenv("OMPCLUSTER_SCHEDULER");
  std::string sched_str = (str != nullptr) ? str : "";

  if (sched_str == "roundrobin") {
    KG_TRACE(10,("[SCHEDULER] Creating dynamic round-robin scheduler\n"));
    scheduler = new RoundRobinScheduler();
  } else if (sched_str == "graph_roundrobin") {
    KG_TRACE(10,("[SCHEDULER] Creating static round-robin scheduler\n"));
    scheduler = new GraphRoundRobinScheduler();
  } else if (sched_str == "heft") {
    KG_TRACE(10,("[SCHEDULER] Creating HEFT scheduler\n"));
    scheduler = new HEFTScheduler();
  } else {
    if (sched_str != "") {
      WarnUnknownScheduler(sched_str);
    }
    KG_TRACE(10,("[SCHEDULER] Creating default scheduler\n"));
    scheduler = new DefaultScheduler();
  }
}

void __kmpc_DeinitScheduler() {
  if (scheduler != nullptr) {
    KG_TRACE(10,("[SCHEDULER] Deinit scheduler\n"));
    delete scheduler;
  }
}

void __kmpc_scheduleGraph(kmp_tdg_info_t* tdg) {
KG_TRACE(10,("__kmpc_scheduleGraph Start\n"));
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  kmp_int gtid = __kmp_get_gtid();

  KG_TRACE(10,("__kmpc_scheduleGraph : thread %d scheduling\n", gtid));
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->acquireAndScheduleGraph(tdg);
  KG_TRACE(10,("__kmpc_scheduleGraph Exit: End Schedule\n"));
  return;
}

bool __kmpc_is_rescheduling(){
  return scheduler->is_rescheduling();  
}

void __kmpc_waitScheduleGraph(){
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }
  //KG_TRACE(10,("__kmpc_waitScheduleGraph Start\n"));
  scheduler->waitScheduleGraph();
  //KG_TRACE(10,("__kmpc_waitScheduleGraph Exit End Wait\n"));
}

int32_t __kmpc_scheduleTask(int32_t task_id) {
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }
  BaseScheduler::LockGuard SchLk(scheduler);
  return scheduler->scheduleTask(task_id);
}

kmp_int32 __kmpc_get_task_device(kmp_int32 task_id) {
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }
  BaseScheduler::LockGuard SchLk(scheduler);
  return scheduler->get_scheduleTask(task_id);
}

void __kmpc_pinTask(int32_t task_id, int DeviceId) {
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->pinTask(task_id, DeviceId);
}

void __kmpc_setTaskState(int32_t task_id, TaskState new_state) {
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->setTaskState(task_id, new_state);
}

void __kmpc_setTaskStateWithDevice(int32_t task_id, int device_id,
                                   TaskState new_state) {
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->setTaskState(task_id, device_id, new_state);
}

kmp_target_dep __kmpc_getDep(int32_t task_id, int64_t addr) {
  return scheduler->getDep(task_id, addr);
}

void __kmpc_dumpTaskGraph() { scheduler->dumpTaskGraph(); }

// fixme: can be simplified?
const void *__kmpc_GetCurrentOutlinedFnID() {
  if (scheduler == nullptr)
    return nullptr;
  BaseScheduler::LockGuard SchLk(scheduler);
  const int CurrTaskID = __kmpc_target_task_current_task_id();
  TaskScheduleData ScheduleData = scheduler->getTask(CurrTaskID);

  if (!ScheduleData.task)
    return nullptr;
  return ScheduleData.task->target_data->outlined_fn_id;
}

void __kmpc_reset_scheduler_barrier(){
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  } 
 BaseScheduler::LockGuard SchLk(scheduler);
 scheduler->setIsScheduleDone(0);
 scheduler->resetSchedulingThread();
}

void __kmpc_allow_exec(){
 assert(scheduler != nullptr);
 BaseScheduler::LockGuard SchLk(scheduler);
 scheduler->setIsScheduleDone(1);
}

void __kmpc_reset_schedule_map(kmp_tdg_info_t* tdg){
  if (scheduler == nullptr)
    return ;
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->resetTasksState();
} 

void __kmpc_scheduler_update_weights(){
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  } 
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->resetWeight();
  scheduler->resetTasksRank();
  scheduler->resetMemoryTasks();
}

void __kmpc_print_profiled_comp_cost(kmp_tdg_info_t* tdg){
  for ( kmp_int32 tdg_task_id = 0; tdg_task_id < tdg->num_tasks ; tdg_task_id++){
     printf("[Inter HEFT]: Task %d:\n",tdg_task_id);
    for (kmp_int32 device = 0 ; device < __kmpc_get_num_devices(); device++) {
     printf("device %d : value: %d ; iteration number: %d ",device, tdg->record_map[tdg_task_id].task_time[device].mean_time,tdg->record_map[tdg_task_id].task_time[device].iteration_number);
    } 
     printf("\n");
  }
}


