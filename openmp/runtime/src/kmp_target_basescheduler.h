//===--------------- basescheduler.h - Target-Task scheduler --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the base abstract scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_BASESCHEDULER_H
#define _OMPTARGET_BASESCHEDULER_H

#include "kmp.h"
#include <cassert>
#include <cstdint>
#include <cstdio>

#include <list>
#include <vector>
#include <set>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <deque>

#define assertm(exp, msg) assert(((void)msg, exp));

#define KMP_TASKNODE_TO_TASKID(Task) KMP_TASK_TO_TASKDATA(Task.task)->td_task_id
#define KMP_TASKPTR_TO_TASKID(Task) KMP_TASK_TO_TASKDATA(Task->task)->td_task_id
#define KMP_TASKNODE_TO_TDG_TASKID(Task) KMP_TASK_TO_TASKDATA(Task.task)->td_tdg_task_id
#define KMP_TASKPTR_TO_TDG_TASKID(Task) KMP_TASK_TO_TASKDATA(Task->task)->td_tdg_task_id
#define KMP_TDG_TASKID_TO_TASKID(tdg,tdg_task_id) KMP_TASKNODE_TO_TASKID(tdg->record_map[tdg_task_id])

enum class NonTaskScheduler {
  STATIC, // Default strategy
  ROUNDROBIN
};

// OmpCluster Task Graph implementation
// =============================================================================

class BaseScheduler {
private:
  std::atomic<int> remaining_target_tasks;

  // Info for non-nowait target round-robin scheduling.
  std::atomic<size_t> non_nowait_counter;

  NonTaskScheduler non_task_scheduler;

public:
  // Internal scheduler mutex
  std::mutex scheduler_mtx;

protected:
  // Info for task-graph dumping.
  std::string graph_dump_dir_path;
  kmp_int32 graph_counter;

  kmp_int32 schedulingThread = -1;
  std::atomic<bool> isScheduleDone;
  std::atomic<bool> hasAcquiredGraph{false};
  kmp_tdg_info *tdg;

  // Scheduling policies
  kmp_int32 initial_schedule_policy = 0;
  kmp_int32 initial_iteration = 0;
  kmp_int32 rescheduling_interval = 0;
  kmp_int32 scheduling_iteration_counter = 0;

public:
  BaseScheduler();
  virtual ~BaseScheduler();

  void acquireAndScheduleGraph(kmp_tdg_info_t* tdg);
  void acquireAndScheduleGraphRR(kmp_tdg_info_t* tdg, kmp_int32 offset);
  void acquireGraphOnly(kmp_tdg_info_t *tdg);
  kmp_int32 scheduleTask(kmp_int32 task_id);
  kmp_int32 get_scheduleTask(kmp_int32 task_id);
  void pinTask(kmp_int32 task_id, kmp_int32 DeviceId);
  void setTaskState(kmp_int32 task_id, TaskState new_state);
  void setTaskState(kmp_int32 task_id, kmp_int32 device_id, TaskState new_state);
  //void setTaskSourceLocation(int32_t task_id, ident_t *loc);

  bool areTargetTaskRemaining() { return remaining_target_tasks > 0; }
  kmp_target_dep getDep(kmp_int32 task_id, kmp_int64 addr);
  TaskScheduleData getTask(kmp_int32 task_id);

  void waitScheduleGraph(kmp_int32 gtid);
  void setIsScheduleDone(bool state);
  void setSchedulingThread(kmp_int32 gtid);
  void resetSchedulingThread();
  bool isSchedulingThread(kmp_int32 gtid); 
  void resetTasksState();
  void resetMemoryTasks();
  void resetTasksRank();
  void scheduleGraphRR(kmp_int32 offset);
  bool is_rescheduling();
  kmp_int32 inc_scheduling_iteration_counter();
  kmp_int32 get_scheduling_iteration_counter();
  void addMemoryTasks();

private:
  /// Replace the current task graph with the newest one from OpenMP runtime.
  void acquireNewGraph(kmp_tdg_info_t* tdg);

public:
  /// Dump the task graph to a graphviz "dot" file.
  virtual void dumpTaskGraph();
  virtual void resetWeight() = 0;

private:
  /// Let subclasses define custom behavior after a new graph is acquired.
  virtual void acquireNewGraphPost() = 0;

  /// Schedule a task graph.
  virtual void scheduleGraph() = 0;

  /// Let subclasses define custom behavior before a task schedule is acquired.
  ///
  /// This function can be useful for schedulers that do not schedule the whole
  /// graph prior to the tasks execution or for the ones that modify the graph
  /// scheduling at task execution time.
  virtual void preScheduleTask(kmp_int32 task_schedule, kmp_int32) = 0;

public:
  struct LockGuard {
    BaseScheduler *sch;

  public:
    LockGuard(BaseScheduler *sch) : sch(sch) { sch->scheduler_mtx.lock(); }
    ~LockGuard() { sch->scheduler_mtx.unlock(); }
  };
};

using TaskPtr= const kmp_node_info*;

namespace detail {

/// Return a string indicating which color an edge should be in the task graph
/// dump.
static inline const char *getEdgeColor(kmp_int32 Rank1,kmp_int32 Rank2) {
  if (Rank1 == -1 || Rank2 == -1)
    return "black";
  if (Rank1 == Rank2)
    return "blue";
  return "red";
}

} // namespace detail

// Task Predicates
// =============================================================================

/// Predicate that indicates if the task is a root task
static inline bool isRootTask(const kmp_node_info &task) {
  return !task.npredecessors;
}

/// Predicate that indicates if the task is a root task
static inline bool isRootTask(TaskPtr task) {
  assert(task != nullptr);
  return isRootTask(*task);
}

/// Predicate that indicates if the task is an exit task
static inline bool isExitTask(const kmp_node_info &task) {
  return !task.nsuccessors;
}

/// Predicate that indicates if the task is an exit task
static inline bool isExitTask(TaskPtr task) {
  assert(task != nullptr);
  return isExitTask(*task);
}

/// Predicate that indicates if the task is a target data task
///
/// These tasks are inserted during codegen to manipulate data
/// (alloc/dealloc/copy) but does not properly execute an outlined function.
static inline bool isTargetDataTask(const kmp_node_info &task) {
  return task.is_target_task && task.target_data->outlined_fn_id == nullptr;
}

/// Predicate that indicates if the task is a target data task
static inline bool isTargetDataTask(TaskPtr task) {
  assert(task != nullptr);
  return isTargetDataTask(*task);
}

/// Predicate that indicates if the task is a classical task
static inline bool isClassicalTask(const kmp_node_info &task) {
  return !task.is_target_task;
}

/// Predicate that indicates if the task is a classical task
static inline bool isClassicalTask(TaskPtr task) {
  assert(task != nullptr);
  return isClassicalTask(*task);
}

/// Predicate that indicates if the task has an user-specified device
static inline bool hasSpecifiedDevice(const kmp_node_info &task) {
  return task.target_data->device_id >= 0;
}

/// Predicate that indicates if the task has an user-specified device
static inline bool hasSpecifiedDevice(TaskPtr task) {
  assert(task != nullptr);
  return hasSpecifiedDevice(*task);
}

// Task Iterators
// =============================================================================

/// A simple iterator for plain arrays.
///
/// This iterator works with both const and non-const pointers.
template <typename T> class RangeIterator {
public:
  RangeIterator(T *ptr, size_t size) : ptr(ptr), size(size) {}

  T *begin() { return &ptr[0]; }
  T *end() { return &ptr[size]; }

  const T *cbegin() const { return &ptr[0]; }
  const T *cend() const { return &ptr[size]; }

public:
  T *ptr;
  size_t size;
};

/// Sort the tasks in topological order and return a list.
///
/// \note This is an implementation of Kahn's algorithm.
/// \see https://en.wikipedia.org/wiki/Topological_sorting
std::list<TaskPtr> TopologicalOrder(kmp_tdg_info &tdg);

/// Sort the tasks in topological order and return a list.
std::list<TaskPtr> TopologicalOrder(kmp_tdg_info *tdg);

/// Sort the tasks in reverse topological order and return a list.
///
/// \see TopologicalOrder
std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info &tdg);

/// Sort the tasks in reverse topological order and return a list.
///
/// \see TopologicalOrder
std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info *tdg);

#endif /* _OMPTARGET_BASESCHEDULER_H */
