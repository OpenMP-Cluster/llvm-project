//===----------------------- kmp_target_heft.h - HEFT scheduler
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heterogeneous Earliest Finish Time algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_HEFT_H
#define _OMPTARGET_HEFT_H

#include "kmp_target_basescheduler.h"
#include "kmp.h" 

#include <deque>

// Forward declaration
class HEFTScheduler;

namespace detail {

// Types
// =============================================================================

/// Priority queue, ordered from the higher upward rank to the lowest.
using PriorityQueue = std::deque<std::pair<TaskPtr, double>>;

// Classes
// =============================================================================

/// Identifies which rank, thread pair the task should be executed at.
struct DeviceId {
  /// The MPI rank of the worker node.
  int32_t Rank;
  /// The ID of the event handler thread.
  int32_t Thread;

  explicit DeviceId(int32_t Rank, int32_t Thread = -1)
      : Rank(Rank), Thread(Thread) {}

  explicit DeviceId(const TaskScheduleData &Data)
      : Rank(Data.rank), Thread(Data.thread) {}

  ~DeviceId() = default;

  static DeviceId head() { return DeviceId{0, -1}; }

  bool operator==(const DeviceId &Other) const {
    return Rank == Other.Rank && Thread == Other.Thread;
  }

  /// Hashing function for this type so we can use it as unordered map keys
  struct Hash {
    size_t operator()(const DeviceId &Device) const {
      return std::hash<int32_t>()(Device.Rank) ^
             std::hash<int32_t>()(Device.Thread);
    }
  };
};

/// Context of the HEFT scheduler.
///
/// This is the class that does all of the heavy lifting in terms of computing
/// the intermediate values for the HEFT algorithm. For clarification about what
/// each of these data structures mean, refer to the original HEFT paper.
///
/// \note As an implementation detail, this class makes heavy use of
/// `std::unordered_map` (hash) maps because of its almost-constant lookup time,
/// in constrast to `std::map`'s logarithm lookup.
///
/// \see HEFTScheduler
/// \see https://ieeexplore.ieee.org/document/993206
class HEFTContext {
public:
  /// Map a (task, device) pair to a timepoint.
  ///
  /// This table is filled by the scheduler in order to find the device which
  /// minimizes the earliest finish time.
  using EarliestTimeMap =
      std::unordered_map<TaskPtr,
                         std::unordered_map<DeviceId, double, DeviceId::Hash>>;

  /// Maps a task to a timepoint.
  ///
  /// This table indicates the actual start or finish time of a task after being
  /// scheduled.
  using ActualTimeMap = std::unordered_map<TaskPtr, double>;

  /// Device availability map.
  ///
  /// Maps a device ID to the next timepoint it will be available for executing
  /// a new task.
  using DeviceAvailMap = std::unordered_map<DeviceId, double, DeviceId::Hash>;

  /// Upward rank map.
  ///
  /// Maps a task to its upward rank priority.
  using UpwardRankMap = std::unordered_map<TaskPtr, double>;

  /// Communication map.
  ///
  /// Indicates how many bytes should be transferred from one task to another.
  using CommunicationMap =
      std::unordered_map<TaskPtr, std::unordered_map<TaskPtr, double>>;

  /// Computation map.
  ///
  /// Indicates how long each task should take to execute after being scheduled.
  using AverageComputationMap = std::unordered_map<TaskPtr, double>;
  
  /// Computation map.
  ///
  /// Indicates how long each task should take to execute after being scheduled on each device.
  using ComputationMap = std::unordered_map<TaskPtr,std::unordered_map<kmp_int32,double>>;

  friend class ::HEFTScheduler;

public:
  /// Default constructor.
  HEFTContext(ScheduleMap &Schedule, kmp_tdg_info *tdg);

  /// Destructor.
  ~HEFTContext() = default;

  /// Compute upward rank for every task in the graph.
  ///
  /// This should be called **before** starting the scheduling process.
  void computeUpwardRanks();

  /// Compute the earliest finish time for a specific (task, rank) pair.
  ///
  /// \returns the computed EFT.
  double computeEFT(TaskPtr Task, DeviceId Device);

  /// Schedule the task on the best rank available based on the EFT
  /// calculations.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTaskOnBestRank(TaskPtr Task);

  /// Force a task to be scheduled on a specific rank irrespective its
  /// predecessors.
  ///
  /// \param task the task
  /// \param rank the rank
  void forceScheduleTask(TaskPtr Task, DeviceId Device);

  /// Defer scheduling of task for later.
  ///
  /// \param task the task
  void deferTask(TaskPtr Task);

  /// Returns a boolean indicating if this task is deferred.
  ///
  /// \param task the task
  inline bool isDeferredTask(const kmp_node_info &Task) const {
    bool failed = isTargetDataTask(Task) &&
                  Schedule[KMP_TASKNODE_TO_TASKID(Task)].state == TaskState::FAILED;
    return Schedule[KMP_TASKNODE_TO_TASKID(Task)].state == TaskState::DEFERRED || failed;
  }

  /// Build a priority queue from the upward rank map.
  ///
  /// \returns the priority queue built.
  PriorityQueue buildQueue() const;

  /// Build a list of device ids from the total number of ranks and handler
  /// threads.
  std::vector<DeviceId> buildDeviceList(int32_t NumRanks) const;

  /// Clear the structures used by the context.
  void clear();

  void clearCompCost();

  /// Construct CompCost Matrix from profiled informations
  void extrapolateCompCost(kmp_int32 scheduling_iteration_counter, kmp_int32 scheduling_policy, kmp_int32 initial_iteration);
 
  /// Set a new task map.
  void setTaskMap(int32_t tdg_idx);

  // Dump internal HEFT information to a file.
  void dump(int32_t NumDevices);

private:
  void computeAvgCompCosts();
  void homogeneousInterpolation();
  void fastProportionalInterpolation();
  void generalProportionalInterpolation();
  /// Compute the earliest start time for a specific (task, rank) pair.
  ///
  /// \returns the computed EST.
  double computeEST(TaskPtr Task, DeviceId Device);

  /// Commit a task to run on a specific rank.
  ///
  /// The task must not be rescheduled after commit.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTask(TaskPtr Task, DeviceId Device);

  /// Estimate the communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param RankA the rank where A's data should read from
  /// \param B the successor task
  /// \param RankB the rank where B's data should write to
  ///
  /// \returns the communication cost from A -> B.
  double computeCommCost(const kmp_node_info &A, DeviceId DeviceA,
                         const kmp_node_info &B, DeviceId DeviceB);

  /// Estimate the average communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param B the successor task
  ///
  /// \returns the average communication cost from A -> B
  double computeAvgCommCost(const kmp_node_info &A,
                            const kmp_node_info &B);

  /// Estimate the average computation cost of a task.
  ///
  /// \param task the task
  ///
  /// \retunrs the average computation cost
  double computeAvgCompCost(const kmp_node_info &Task);

  /// Estimate the average computation cost of a task running on a specific
  /// rank.
  ///
  /// \param task the task
  /// \param trank the rank of the task
  ///
  /// \retunrs the average computation cost
  double computeCompCost(const kmp_node_info &Task, DeviceId Device);

private:
  /// Coefficient of the communication cost.
  /// Having a higher coefficient means that communication will have more weight
  /// in relation to computation.
  double CommMultiplier;
  /// Default computation cost of tasks
  double DefaultCompCost;
  /// Default communication cost of tasks
  double DefaultCommCost;
  /// Number of threads in each node to choose from.
  int NumThreads;
  /// List of tasks to be scheduled
  kmp_tdg_info *tdg;
  /// State of the scheduler
  ScheduleMap &Schedule;
  /// Earliest start time
  EarliestTimeMap EST;
  /// Earliest finish time
  EarliestTimeMap EFT;
  /// Actual start time
  ActualTimeMap AST;
  /// Actual finish time
  ActualTimeMap AFT;
  /// Upward ranks
  UpwardRankMap UpwardRank;
  /// Device availability
  DeviceAvailMap DeviceAvail;
  /// Communication map
  CommunicationMap CommunicationCost;
  /// Average Computation map
  AverageComputationMap AverageComputationCost;
  /// Computation map
  ComputationMap ComputationCost;
  /// File where to dump HEFT information
  std::string HEFTLogPath;
};

} // namespace detail

/// Implementation of the scheduler interface for HEFT.
///
/// This class provides a simple interface that external users can call to run
/// the HEFT scheduler.
///
/// \see HEFTContext
class HEFTScheduler final : public BaseScheduler {
public:
  /// Default constructor.
  HEFTScheduler() : Ctx(this->tdg->schedule_map, this->tdg) {}

  /// Destructor.
  ~HEFTScheduler() {}

public:
  std::unordered_map<int32_t, std::set<int>> dataTaskSuccessorRanks;

private:
  /// Run HEFT scheduling algorithm.
  void scheduleGraph() override;

  /// Clear context after acquiring new graph.
  void acquireNewGraphPost() override;

  /// Do nothing.
  void preScheduleTask(int32_t task_id, int32_t) override {}
  
  void resetWeight() override;

public:
  /// Dump the task graph with HEFT-specific information.
  void dumpTaskGraph() override;

  std::set<int> getDataTaskSuccessors(int32_t task_id) override;

private:
  detail::HEFTContext Ctx;

private:
  struct DumpOptions {
    DumpOptions();
    /// Whether to dump the communication cost in the edges
    bool EdgeLabel = true;
    /// Whether to show predecessors or not
    bool Predecessors = false;
    /// Wether to dump a fake, unique entry node
    bool FakeEntryNode = false;
    /// Wether to dump a fake, unique exit node
    bool FakeExitNode = false;
  };

  DumpOptions Dump;
};

#endif /* _OMPTARGET_HEFT_H */
