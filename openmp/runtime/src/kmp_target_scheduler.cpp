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


void __kmpc_remove_task(kmp_task_t *task , kmp_task_t *predecessor){
  
  kmp_int gtid = __kmp_get_gtid();
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_depnode_t *node = task->td_depnode;
  
  //Removing the task from the successor list of the predecessor
  // We could use a list of predecessors as input to deal with tasks with multiple dependencies. However, it's not useful with memory tasks
  
  kmp_depnode_t *node_pred = predecessor->td_depnode;

  // Check mutexinoutset dependencies, release locks
  if (UNLIKELY(node_pred && (node_pred->dn.mtx_num_locks < 0))) {
    // negative num_locks means all locks were acquired
    node_pred->dn.mtx_num_locks = -node_pred->dn.mtx_num_locks;
    for (int i = node_pred->dn.mtx_num_locks - 1; i >= 0; --i) {
      KMP_DEBUG_ASSERT(node_pred->dn.mtx_locks[i] != NULL);
      __kmp_release_lock(node_pred->dn.mtx_locks[i], gtid);
    }
  }

  
  kmp_depnode_list_t* pred_successors = node_pred->successors;
  kmp_depnode_list_t* pred_successors_next = node_pred->successors->next;

  while ( pred_successors_next->next != nullptr && pred_sucessors_next->task != task) {
    pred_successors = pred_successors_next;
    pred_successors_next = pred_successors->next;
  }

  if (pred_successors_next != nullptr) {
    //Simply remove the link containing the task
    pred_successors->next = pred_successors_next->next;
  } 
  else {
    KG_TRACE(10,"Given predecessor is not a predecessor of the task");
  }

  //Removing the successors
  // Check mutexinoutset dependencies, release locks
  if (UNLIKELY(node && (node->dn.mtx_num_locks < 0))) {
    // negative num_locks means all locks were acquired
    node->dn.mtx_num_locks = -node->dn.mtx_num_locks;
    for (int i = node->dn.mtx_num_locks - 1; i >= 0; --i) {
      KMP_DEBUG_ASSERT(node->dn.mtx_locks[i] != NULL);
      __kmp_release_lock(node->dn.mtx_locks[i], gtid);
    }
  }

  if (task->td_dephash) {
     __kmp_dephash_free(thread, task->td_dephash);
    task->td_dephash = NULL;
  }

  if (!node)
    return;

  KMP_ACQUIRE_DEPNODE(gtid, node);

  node->dn.task =
        NULL; // mark this task as finished, so no new dependencies are generated
  KMP_RELEASE_DEPNODE(gtid, node);

  kmp_depnode_list_t *next;
  kmp_taskdata_t *next_taskdata;
  for (kmp_depnode_list_t *p = node->dn.successors; p; p = next) {
    kmp_depnode_t *successor = p->node;
    kmp_int32 npredecessors = KMP_ATOMIC_DEC(&successor->dn.npredecessors) - 1;
    //We don't want to launch the successor if npredecessors=0, we let the taskgraph do it
    next = p->next;
    __kmp_node_deref(thread, p->node);
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, p);
#else
    __kmp_thread_free(thread, p);
#endif
  }
  __kmp_node_deref(thread, node);

#if USE_FAST_MEMORY
  __kmp_fast_free(thread, task);
  __kmp_fast_free(thread, taskdata);
#else /* ! USE_FAST_MEMORY */
  __kmp_thread_free(thread, task);
  __kmp_thread_free(thread, taskdata);
#endif
}
