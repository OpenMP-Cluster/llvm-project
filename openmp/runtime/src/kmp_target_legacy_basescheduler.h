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

#ifndef _OMPTARGET_LEGACY_BASESCHEDULER_H
#define _OMPTARGET_LEGACY_BASESCHEDULER_H

// #include "SourceInfo.h"

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

// fixme: find a better way to define this
//  Debugging messages
#ifdef OMPTARGET_DEBUG
#include <stdio.h>

#define DEBUGP(prefix, ...)                                                    \
  {                                                                            \
    fprintf(stderr, "%s --> ", prefix);                                        \
    fprintf(stderr, __VA_ARGS__);                                              \
  }

/// Emit a message for debugging
#define DP(...)                                                                \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DEBUGP(DEBUG_PREFIX, __VA_ARGS__);                                       \
    }                                                                          \
  } while (false)

/// Emit a message for debugging or failure if debugging is disabled
#define REPORT(...)                                                            \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DP(__VA_ARGS__);                                                         \
    } else {                                                                   \
      FAILURE_MESSAGE(__VA_ARGS__);                                            \
    }                                                                          \
  } while (false)
#else
#define DEBUGP(prefix, ...)                                                    \
  {}
#define DP(...)                                                                \
  {}
#define REPORT(...) FAILURE_MESSAGE(__VA_ARGS__);
#endif // OMPTARGET_DEBUG

enum class LegacyNonTaskScheduler {
  STATIC, // Default strategy
  ROUNDROBIN
};

// OmpCluster Task Graph implementation
// =============================================================================
enum class LegacyTaskState {
  UNDEFINED = -1,
  CREATED,
  DEFERRED, // Only used by HEFTScheduler
  SCHEDULED,
  DISPATCHED,
  EXECUTED,
  FAILED,
  SAVED
};

struct LegacyTaskScheduleData {
  LegacyTaskState state;
  int rank;
  int thread;
  int translated_rank;
  const kmp_target_task_t *task;
  std::string source_location;
};

// Maps a task ID to its associated schedule data
using LegacyScheduleMap = std::unordered_map<int32_t, LegacyTaskScheduleData>;

class LegacyBaseScheduler {
private:
  int remaining_target_tasks;

  // Info for non-nowait target round-robin scheduling.
  std::atomic<size_t> non_nowait_counter;

  LegacyNonTaskScheduler non_task_scheduler;
public:
  // Internal scheduler mutex
  std::mutex scheduler_mtx;

private:
  bool isScheduleDone;
  int32_t schedulingThread;

protected:
  // Info for task-graph dumping.
  std::string graph_dump_dir_path;
  int graph_counter;

  LegacyScheduleMap schedule_map;
  kmp_target_task_map_t *task_map;

public:
  LegacyBaseScheduler();
  virtual ~LegacyBaseScheduler();
  void acquireAndScheduleGraph();
  int32_t scheduleTask(int32_t task_id);
  void pinTask(int32_t task_id, int LegacyDeviceId);
  void setTaskState(int32_t task_id, LegacyTaskState new_state);
  void setTaskState(int32_t task_id, int device_id, LegacyTaskState new_state);
  //  void setTaskSourceLocation(int32_t task_id, ident_t *loc);
  bool areTargetTaskRemaining() { return remaining_target_tasks > 0; }
  kmp_target_dep getDep(int32_t task_id, int64_t addr);
  LegacyTaskScheduleData getTask(int task_id);
  bool isTargetDataTask(int32_t task_id);
  bool waitScheduleGraph();
  void setIsScheduleDone(bool state);
  void setSchedulingThread(int32_t gtid);
  void resetSchedulingThread();
  bool isSchedulingThread(int32_t gtid);

private:
  /// Wait for OpenMP to create its graph
  void waitOmpGraphCreation();
  /// Replace the current task graph with the newest one from OpenMP runtime.
  void acquireNewGraph();

public:
  /// Dump the task graph to a graphviz "dot" file.
  virtual void dumpTaskGraph();
  /// Return the list of successor ranks of a given data task if the HEFT
  /// scheduler is enabled
  virtual std::set<int> getDataTaskSuccessors(int32_t task_id) { return {}; }

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
  virtual void preScheduleTask(int32_t task_schedule, int32_t) = 0;

public:
  struct LockGuard {
    LegacyBaseScheduler *sch;

  public:
    LockGuard(LegacyBaseScheduler *sch) : sch(sch) { sch->scheduler_mtx.lock(); }
    ~LockGuard() { sch->scheduler_mtx.unlock(); }
  };
};

using LegacyTaskPtr = const kmp_target_task_t *;

namespace detail {

/// Return a string indicating which color an edge should be in the task graph
/// dump.
static inline const char *getLegacyEdgeColor(int32_t Rank1, int32_t Rank2) {
  if (Rank1 == -1 || Rank2 == -1)
    return "black";
  if (Rank1 == Rank2)
    return "blue";
  return "red";
}

} // namespace detail

// Task Utilitaries
// =============================================================================

static inline int32_t NumSuccessors(const kmp_target_task_t &task) {
  int32_t count = 0;
  for (auto *ptr = task.successors; ptr != nullptr; ptr = ptr->next)
    count++;
  return count;
}

static inline int32_t NumSuccessors(const kmp_target_task_t *task) {
  assert(task != nullptr);
  return NumSuccessors(*task);
}

static inline int32_t NumPredecessors(const kmp_target_task_t &task) {
  return task.num_predecessors;
}

static inline int32_t NumPredecessors(const kmp_target_task_t *task) {
  assert(task != nullptr);
  return NumPredecessors(*task);
}

// Task Predicates
// =============================================================================

/// Predicate that indicates if the task is a root task
static inline bool isLegacyRootTask(const kmp_target_task_t &task) {
  return task.predecessors == nullptr;
}

/// Predicate that indicates if the task is a root task
static inline bool isLegacyRootTask(LegacyTaskPtr task) {
  assert(task != nullptr);
  return isLegacyRootTask(*task);
}

/// Predicate that indicates if the task is an exit task
static inline bool isLegacyExitTask(const kmp_target_task_t &task) {
  return task.successors == nullptr;
}

/// Predicate that indicates if the task is an exit task
static inline bool isLegacyExitTask(LegacyTaskPtr task) {
  assert(task != nullptr);
  return isLegacyExitTask(*task);
}

/// Predicate that indicates if the task is a target data task
///
/// These tasks are inserted during codegen to manipulate data
/// (alloc/dealloc/copy) but does not properly execute an outlined function.
static inline bool isLegacyTargetDataTask(const kmp_target_task_t &task) {
  return task.is_target_task && task.target_data.outlined_fn_id == nullptr;
}

/// Predicate that indicates if the task is a target data task
static inline bool isLegacyTargetDataTask(LegacyTaskPtr task) {
  assert(task != nullptr);
  return isLegacyTargetDataTask(*task);
}

/// Predicate that indicates if the task is a classical task
static inline bool isLegacyClassicalTask(const kmp_target_task_t &task) {
  return !task.is_target_task;
}

/// Predicate that indicates if the task is a classical task
static inline bool isLegacyClassicalTask(LegacyTaskPtr task) {
  assert(task != nullptr);
  return isLegacyClassicalTask(*task);
}

// Task Iterators
// =============================================================================

/// A simple iterator for plain arrays.
///
/// This iterator works with both const and non-const pointers.
template <typename T> class LegacyRangeIterator {
public:
  LegacyRangeIterator(T *ptr, size_t size) : ptr(ptr), size(size) {}

  T *begin() { return &ptr[0]; }
  T *end() { return &ptr[size]; }

  const T *cbegin() const { return &ptr[0]; }
  const T *cend() const { return &ptr[size]; }

public:
  T *ptr;
  size_t size;
};

// Type aliases
using TaskIterator = LegacyRangeIterator<kmp_target_task_t>;
using RootTaskIterator = LegacyRangeIterator<kmp_target_task_t *>;

/// Create an iterator for the tasks in a task map.
static inline TaskIterator Tasks(kmp_target_task_map_t &TaskMap) {
  return {TaskMap.tasks, (size_t)TaskMap.num_tasks};
}

/// Create an iterator for the tasks in a task map.
static inline TaskIterator Tasks(kmp_target_task_map_t *TaskMap) {
  assert(TaskMap != nullptr);
  return Tasks(*TaskMap);
}

/// Create an iterator for the root tasks in a task map.
static inline RootTaskIterator Roots(kmp_target_task_map_t &TaskMap) {
  return {TaskMap.roots, (size_t)TaskMap.num_roots};
}

/// Create an iterator for the root tasks in a task map.
static inline RootTaskIterator Roots(kmp_target_task_map_t *TaskMap) {
  assert(TaskMap != nullptr);
  return Roots(*TaskMap);
}

/// An iterator for a linked list of tasks.
///
/// This class is just a front-end for the actual Iterator that is implemented
/// as a nested class. This class will be useful for iterating over
/// successors/predecessors of a task.
///
/// FIXME: Is there a way of not duplicating Iterator/ConstIterator?
class TaskListIterator {
public:
  TaskListIterator(kmp_target_task_list_node_t *node) : m_RootNode(node) {}

  // Forward declarations
  class Iterator;
  class ConstIterator;

  Iterator begin() { return {m_RootNode}; }
  Iterator end() { return {nullptr}; }

  ConstIterator cbegin() const { return {m_RootNode}; }
  ConstIterator cend() const { return {nullptr}; }

  /// Iterator for mutable list elements
  class Iterator {
  public:
    Iterator(kmp_target_task_list_node_t *node) : m_CurrentNode(node) {}

    /// Advance iterator to next element in the list
    Iterator &operator++() {
      if (m_CurrentNode)
        m_CurrentNode = m_CurrentNode->next;
      return *this;
    }

    /// Compare iterators
    bool operator!=(const Iterator &iter) {
      return m_CurrentNode != iter.m_CurrentNode;
    }

    /// Dereference iterator to obtain list element
    kmp_target_task_t &operator*() { return *m_CurrentNode->task; }

  private:
    /// Current node in the list
    kmp_target_task_list_node_t *m_CurrentNode;
  };

  /// Iterator for immutable list elements
  class ConstIterator {
  public:
    ConstIterator(kmp_target_task_list_node_t const *node)
        : m_CurrentNode(node) {}

    /// Advance iterator to next element in the list
    ConstIterator &operator++() {
      if (m_CurrentNode)
        m_CurrentNode = m_CurrentNode->next;
      return *this;
    }

    /// Compare iterators
    bool operator!=(const ConstIterator &iter) {
      return m_CurrentNode != iter.m_CurrentNode;
    }

    /// Dereference iterator to obtain list element
    const kmp_target_task_t &operator*() { return *m_CurrentNode->task; }

  private:
    /// Current node in the list
    kmp_target_task_list_node_t const *m_CurrentNode;
  };

private:
  /// Root node (first) of the list
  kmp_target_task_list_node_t *m_RootNode;
};

/// Create an iterator for the successors of a task.
static inline TaskListIterator Successors(const kmp_target_task_t &Task) {
  return TaskListIterator{Task.successors};
}

/// Create an iterator for the successors of a task.
static inline TaskListIterator Successors(const kmp_target_task_t *Task) {
  assert(Task != nullptr);
  return Successors(*Task);
}

/// Create an iterator for the predecessors of a task.
static inline TaskListIterator Predecessors(const kmp_target_task_t &Task) {
  return TaskListIterator{Task.predecessors};
}

/// Create an iterator for the predecessors of a task.
static inline TaskListIterator Predecessors(const kmp_target_task_t *Task) {
  assert(Task != nullptr);
  return Predecessors(*Task);
}

/// Sort the tasks in topological order and return a list.
///
/// \note This is an implementation of Kahn's algorithm.
/// \see https://en.wikipedia.org/wiki/Topological_sorting
std::list<LegacyTaskPtr> TopologicalOrder(kmp_target_task_map_t &TaskMap);

/// Sort the tasks in topological order and return a list.
std::list<LegacyTaskPtr> TopologicalOrder(kmp_target_task_map_t *TaskMap);

/// Sort the tasks in reverse topological order and return a list.
///
/// \see TopologicalOrder
std::list<LegacyTaskPtr> ReverseTopologicalOrder(kmp_target_task_map_t &TaskMap);

/// Sort the tasks in reverse topological order and return a list.
///
/// \see TopologicalOrder
std::list<LegacyTaskPtr> ReverseTopologicalOrder(kmp_target_task_map_t *TaskMap);

#endif /* _OMPTARGET_LEGACY_BASESCHEDULER_H */
