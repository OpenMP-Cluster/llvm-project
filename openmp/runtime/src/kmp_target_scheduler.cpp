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
#include "kmp_target_legacy_heft.h"

static kmp_int32 num_devices = -1;

kmp_int32 __kmpc_get_num_devices(){
  if (num_devices!=-1)
    return num_devices;
  return num_devices = omp_get_num_devices();
}

// Strings allowed to be passed in the OMPCLUSTER_SCHEDULER environment
// variable.
// NOTE: Update this when adding a new scheduler.
static const char *allowed_schedulers[] = {
    "roundrobin",
    "graph_roundrobin",
    "heft",
    "heft_legacy",
};

// Scheduler global state
// static BaseScheduler *scheduler;
static BaseScheduler *scheduler = nullptr;
static LegacyBaseScheduler *legacy_scheduler = nullptr;
static bool use_legacy_scheduler = false;

// Schedule Map for each tdg

static std::unordered_map<kmp_int32, ScheduleMap*> *schedule_map_list;

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

  if (scheduler != nullptr || legacy_scheduler != nullptr) {
    return;
  }
   
  schedule_map_list = new std::unordered_map<kmp_int32, ScheduleMap*>();

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
  } else if (sched_str == "heft_legacy") {
    KG_TRACE(10,("[SCHEDULER] Creating legacy HEFT scheduler\n"));
    use_legacy_scheduler = true;
    legacy_scheduler = new LegacyHEFTScheduler();
  } else {
    if (sched_str != "") {
      WarnUnknownScheduler(sched_str);
    }
    KG_TRACE(10,("[SCHEDULER] Creating default scheduler\n"));
    scheduler = new DefaultScheduler();
  }
}

void __kmpc_DeinitScheduler() {
  if (legacy_scheduler != nullptr) {
    KG_TRACE(10,("[SCHEDULER] Deinit legacy scheduler\n"));
    delete legacy_scheduler;
    legacy_scheduler = nullptr;
  }

  if (scheduler != nullptr) {
    KG_TRACE(10,("[SCHEDULER] Deinit scheduler\n"));
    delete scheduler;
    scheduler = nullptr;
  }

  use_legacy_scheduler = false;
}

bool __kmpc_is_legacy_scheduler() {
  if (scheduler == nullptr && legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  return use_legacy_scheduler;
}

bool __kmpc_legacy_is_target_data_task(int32_t task_id) {
  if (!use_legacy_scheduler || legacy_scheduler == nullptr)
    return false;

  return legacy_scheduler->isTargetDataTask(task_id);
}

void __kmpc_scheduleGraph(kmp_tdg_info_t* tdg) {
KG_TRACE(10,("__kmpc_scheduleGraph Start\n"));
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  kmp_int gtid = __kmp_get_gtid();

  KG_TRACE(30,("__kmpc_scheduleGraph : thread %d scheduling\n", gtid));
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->acquireAndScheduleGraph(tdg);
  KG_TRACE(10,("__kmpc_scheduleGraph Exit: End Schedule\n"));
  return;
}

void __kmpc_scheduleLegacyGraph() {

  if (legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  if (!__kmpc_has_active_tasks()) {
  if (use_legacy_scheduler && legacy_scheduler != nullptr) {
    legacy_scheduler->scheduler_mtx.lock();

    if (!__kmpc_has_active_tasks())
      legacy_scheduler->resetSchedulingThread();

    legacy_scheduler->scheduler_mtx.unlock();
  }

  return;
}

  if (!use_legacy_scheduler || legacy_scheduler == nullptr) {
    return;
  }

  kmp_int gtid = __kmp_get_gtid();

  kmp_int tid = __kmp_get_tid();

  legacy_scheduler->scheduler_mtx.lock();
  legacy_scheduler->setSchedulingThread(gtid);
  bool winner = legacy_scheduler->isSchedulingThread(gtid);
  legacy_scheduler->scheduler_mtx.unlock();



  if (winner) {

    LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
    legacy_scheduler->acquireAndScheduleGraph();

    return;
  }

  legacy_scheduler->waitScheduleGraph();
}

bool __kmpc_is_rescheduling(){
  return scheduler->is_rescheduling();  
}

void __kmpc_waitScheduleGraph(){
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }
  // KG_TRACE(90,("__kmpc_waitScheduleGraph Start\n"));
  scheduler->waitScheduleGraph(__kmp_get_gtid());
  // KG_TRACE(90,("__kmpc_waitScheduleGraph Exit End Wait\n"));
}

int32_t __kmpc_scheduleTask(int32_t task_id) {
  KG_TRACE(90,("__kmpc_scheduleTask Start\n"));

  if (scheduler == nullptr && legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  if (use_legacy_scheduler) {
    LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
    KG_TRACE(90,("__kmpc_scheduleTask End (legacy)\n"));
    return legacy_scheduler->scheduleTask(task_id);
  }

  BaseScheduler::LockGuard SchLk(scheduler);
  KG_TRACE(90,("__kmpc_scheduleTask End\n"));
  return scheduler->scheduleTask(task_id);
}

kmp_int32 __kmpc_get_task_device(kmp_int32 task_id) {
  KG_TRACE(90,("__kmpc_get_task_device Start\n"));
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }
  BaseScheduler::LockGuard SchLk(scheduler);
  KG_TRACE(90,("__kmpc_get_task_device End\n"));
  return scheduler->get_scheduleTask(task_id);
}

void __kmpc_pinTask(int32_t task_id, int DeviceId) {
  KG_TRACE(90,("__kmpc_pinTask Start\n"));

  if (scheduler == nullptr && legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  if (use_legacy_scheduler) {
    LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
    legacy_scheduler->pinTask(task_id, DeviceId);
    KG_TRACE(90,("__kmpc_pinTask End (legacy)\n"));
    return;
  }

  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->pinTask(task_id, DeviceId);
  KG_TRACE(90,("__kmpc_pinTask End\n"));
}

void __kmpc_setTaskState(int32_t task_id, TaskState new_state) {
  KG_TRACE(90,("__kmpc_setTaskState Start\n"));

  if (scheduler == nullptr && legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  if (use_legacy_scheduler) {
    LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
    legacy_scheduler->setTaskState(
        task_id, static_cast<LegacyTaskState>(new_state));
    KG_TRACE(90,("__kmpc_setTaskState End (legacy)\n"));
    return;
  }

  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->setTaskState(task_id, new_state);
  KG_TRACE(90,("__kmpc_setTaskState End\n"));
}

void __kmpc_setTaskStateWithDevice(int32_t task_id, int device_id,
                                   TaskState new_state) {
  KG_TRACE(90,("__kmpc_setTaskStateWithDevice Start\n"));
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->setTaskState(task_id, device_id, new_state);

  KG_TRACE(90,("__kmpc_setTaskStateWithDevice End\n"));
}

void __kmpc_acquireGraphOnly(kmp_tdg_info_t *tdg) {
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  BaseScheduler::LockGuard SchLk(scheduler);



  scheduler->acquireGraphOnly(tdg);
}

kmp_target_dep __kmpc_getDep(int32_t task_id, int64_t addr) {
  KG_TRACE(90,("__kmpc_getDep Start\n"));
  return scheduler->getDep(task_id, addr);
}

void __kmpc_dumpTaskGraph() { scheduler->dumpTaskGraph(); }

// fixme: can be simplified?
const void *__kmpc_GetCurrentOutlinedFnID() {
  KG_TRACE(90,("__kmpc_GetCurrentOutlinedFnID Start\n"));
  if (scheduler == nullptr)
    return nullptr;
  BaseScheduler::LockGuard SchLk(scheduler);
  const int CurrTaskID = __kmpc_target_task_current_task_id();
  TaskScheduleData ScheduleData = scheduler->getTask(CurrTaskID);
  KG_TRACE(90,("__kmpc_GetCurrentOutlinedFnID End\n"));

  if (!ScheduleData.task)
    return nullptr;
  return ScheduleData.task->target_data->outlined_fn_id;
}

void __kmpc_reset_legacy_scheduler_owner() {
  if (!use_legacy_scheduler || legacy_scheduler == nullptr)
    return;

  LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
  legacy_scheduler->resetSchedulingThread();
}

void __kmpc_reset_scheduler_barrier() {
  if (scheduler == nullptr && legacy_scheduler == nullptr) {
    __kmpc_InitScheduler();
  }

  if (use_legacy_scheduler) {

    LegacyBaseScheduler::LockGuard SchLk(legacy_scheduler);
    legacy_scheduler->setIsScheduleDone(false);
    legacy_scheduler->resetSchedulingThread();
    return;
  }

  BaseScheduler::LockGuard SchLk(scheduler);
  // Ordinary OpenMP task-barrier exits must not invalidate a completed
  // scheduling operation. The scheduler manages its own scheduling state.
  scheduler->resetSchedulingThread();
}

void __kmpc_allow_exec(){
 KA_TRACE(90,(" __kmpc_allow_exec Start\n"));
 assert(scheduler != nullptr);
 BaseScheduler::LockGuard SchLk(scheduler);
 scheduler->setIsScheduleDone(1);
 KA_TRACE(90,(" __kmpc_allow_exec End\n"));
}

void __kmpc_reset_schedule_map(kmp_tdg_info_t* tdg){
 KG_TRACE(90,("__kmpc_reset_schedule_map Start\n"));
  if (scheduler == nullptr)
    return ;
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->resetTasksState();
 KG_TRACE(90,("__kmpc_reset_schedule_map End\n"));
} 

void __kmpc_scheduler_update_weights(){
 KG_TRACE(90,(" __kmpc_scheduler_update_weights Start\n"));
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  } 
  BaseScheduler::LockGuard SchLk(scheduler);
  scheduler->resetWeight();
  scheduler->resetTasksRank();
  scheduler->resetMemoryTasks();
 KG_TRACE(90,(" __kmpc_scheduler_update_weights End\n"));
}

void __kmpc_print_profiled_comp_cost(kmp_tdg_info_t* tdg){
//FIXME: This should print in a file 
  for ( kmp_int32 tdg_task_id = 0; tdg_task_id < tdg->num_tasks ; tdg_task_id++){
    KG_TRACE(10,("[Inter HEFT]: Task %d:\n",tdg_task_id));
    for (kmp_int32 device = 0 ; device < __kmpc_get_num_devices(); device++) {
      KG_TRACE(10,("device %d : value: %d ; iteration number: %d ",device, tdg->record_map[tdg_task_id].task_time[device].mean_time,tdg->record_map[tdg_task_id].task_time[device].iteration_number));
    } 
    KG_TRACE(10,("\n"));
  }
}

ScheduleMap* get_schedule_map(int tdg_id) {
 KG_TRACE(90,("*get_schedule_map (enter) tdg_id : %d\n",tdg_id));
  if (scheduler == nullptr) {
    __kmpc_InitScheduler();
  } 
  if (!schedule_map_list->count(tdg_id)) {
    (*schedule_map_list)[tdg_id] = new ScheduleMap();
  }
  KG_TRACE(90,("get_schedule_map (exit) schedule_map address:  : %p\n", schedule_map_list->at(tdg_id)));
  return  (*schedule_map_list)[tdg_id];
}

void __kmpc_free_schedule_maps() {
  // delete __kmp_global_tdgs[0]->scheduling_info->schedule_map;
  //delete __kmp_global_tdgs[0]->scheduling_info;
  // for (auto S: *schedule_map_list) {
  //   delete S.second;
  // }
  // delete schedule_map_list;
}

kmp_int32 __kmpc_get_device_type(kmp_int64 device_id){
  return __tgt_target_get_device_type(device_id);
}

kmp_int32 __kmpc_scheduling_iteration_counter(){
  return scheduler->get_scheduling_iteration_counter();
}

kmp_int32 __kmpc_inc_scheduling_iteration_counter(){
  return scheduler->inc_scheduling_iteration_counter();
}

