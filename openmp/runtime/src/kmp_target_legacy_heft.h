//===----------------------- kmp_target_legacy_heft.h - HEFT scheduler
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

#ifndef _OMPTARGET_LEGACY_HEFT_H
#define _OMPTARGET_LEGACY_HEFT_H

#include "kmp_target_legacy_basescheduler.h"

#include <deque>

// fixme:reimplement usage of profiler info
//  #include "llvm/Support/JSON.h"

// Forward declaration
class LegacyHEFTScheduler;

namespace detail {

// using namespace llvm;

// Types
// =============================================================================

/// Priority queue, ordered from the higher upward rank to the lowest.
using LegacyPriorityQueue = std::deque<std::pair<LegacyTaskPtr, double>>;

// Classes
// =============================================================================

/// Identifies which rank, thread pair the task should be executed at.
struct LegacyDeviceId {
  /// The MPI rank of the worker node.
  int32_t Rank;
  /// The ID of the event handler thread.
  int32_t Thread;

  explicit LegacyDeviceId(int32_t Rank, int32_t Thread = -1)
      : Rank(Rank), Thread(Thread) {}

  explicit LegacyDeviceId(const LegacyTaskScheduleData &Data)
      : Rank(Data.rank), Thread(Data.thread) {}

  ~LegacyDeviceId() = default;

  static LegacyDeviceId head() { return LegacyDeviceId{0, -1}; }

  bool operator==(const LegacyDeviceId &Other) const {
    return Rank == Other.Rank && Thread == Other.Thread;
  }

  /// Hashing function for this type so we can use it as unordered map keys
  struct Hash {
    size_t operator()(const LegacyDeviceId &Device) const {
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
/// \see LegacyHEFTScheduler
/// \see https://ieeexplore.ieee.org/document/993206
class LegacyHEFTContext {
public:
  /// Map a (task, device) pair to a timepoint.
  ///
  /// This table is filled by the scheduler in order to find the device which
  /// minimizes the earliest finish time.
  using EarliestTimeMap =
      std::unordered_map<LegacyTaskPtr,
                         std::unordered_map<LegacyDeviceId, double, LegacyDeviceId::Hash>>;

  /// Maps a task to a timepoint.
  ///
  /// This table indicates the actual start or finish time of a task after being
  /// scheduled.
  using ActualTimeMap = std::unordered_map<LegacyTaskPtr, double>;

  /// Device availability map.
  ///
  /// Maps a device ID to the next timepoint it will be available for executing
  /// a new task.
  using DeviceAvailMap = std::unordered_map<LegacyDeviceId, double, LegacyDeviceId::Hash>;

  /// Upward rank map.
  ///
  /// Maps a task to its upward rank priority.
  using UpwardRankMap = std::unordered_map<LegacyTaskPtr, double>;

  /// Communication map.
  ///
  /// Indicates how many bytes should be transferred from one task to another.
  using CommunicationMap =
      std::unordered_map<LegacyTaskPtr, std::unordered_map<LegacyTaskPtr, double>>;

  /// Computation map.
  ///
  /// Indicates how long each task should take to execute after being scheduled.
  using ComputationMap = std::unordered_map<LegacyTaskPtr, double>;

  friend class ::LegacyHEFTScheduler;

public:
  /// Default constructor.
  LegacyHEFTContext(LegacyScheduleMap &Schedule, kmp_target_task_map_t *TaskMap);

  /// Destructor.
  ~LegacyHEFTContext() = default;

  /// Compute upward rank for every task in the graph.
  ///
  /// This should be called **before** starting the scheduling process.
  void computeUpwardRanks();

  /// Compute the earliest finish time for a specific (task, rank) pair.
  ///
  /// \returns the computed EFT.
  double computeEFT(LegacyTaskPtr Task, LegacyDeviceId Device);

  /// Schedule the task on the best rank available based on the EFT
  /// calculations.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTaskOnBestRank(LegacyTaskPtr Task);

  /// Force a task to be scheduled on a specific rank irrespective its
  /// predecessors.
  ///
  /// \param task the task
  /// \param rank the rank
  void forceScheduleTask(LegacyTaskPtr Task, LegacyDeviceId Device);

  /// Defer scheduling of task for later.
  ///
  /// \param task the task
  void deferTask(LegacyTaskPtr Task);

  /// Returns a boolean indicating if this task is deferred.
  ///
  /// \param task the task
  inline bool isDeferredTask(const kmp_target_task_t &Task) const {
    bool failed = isLegacyTargetDataTask(Task) &&
                  Schedule[Task.task_id].state == LegacyTaskState::FAILED;
    return Schedule[Task.task_id].state == LegacyTaskState::DEFERRED || failed;
  }

  /// Build a priority queue from the upward rank map.
  ///
  /// \returns the priority queue built.
  LegacyPriorityQueue buildQueue() const;

  /// Build a list of device ids from the total number of ranks and handler
  /// threads.
  std::vector<LegacyDeviceId> buildDeviceList(int32_t NumRanks) const;

  /// Clear the structures used by the context.
  void clear();

  /// Set a new task map.
  void setTaskMap(kmp_target_task_map_t *NewTaskMap);

  // Dump internal HEFT information to a file.
  void dump(int32_t NumDevices);

private:
  /// Compute the earliest start time for a specific (task, rank) pair.
  ///
  /// \returns the computed EST.
  double computeEST(LegacyTaskPtr Task, LegacyDeviceId Device);

  /// Commit a task to run on a specific rank.
  ///
  /// The task must not be rescheduled after commit.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTask(LegacyTaskPtr Task, LegacyDeviceId Device);

  /// Estimate the communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param RankA the rank where A's data should read from
  /// \param B the successor task
  /// \param RankB the rank where B's data should write to
  ///
  /// \returns the communication cost from A -> B.
  double computeCommCost(const kmp_target_task_t &A, LegacyDeviceId DeviceA,
                         const kmp_target_task_t &B, LegacyDeviceId DeviceB);

  /// Estimate the average communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param B the successor task
  ///
  /// \returns the average communication cost from A -> B
  double computeAvgCommCost(const kmp_target_task_t &A,
                            const kmp_target_task_t &B);

  /// Estimate the average computation cost of a task.
  ///
  /// \param task the task
  ///
  /// \retunrs the average computation cost
  double computeAvgCompCost(const kmp_target_task_t &Task);

  /// Estimate the average computation cost of a task running on a specific
  /// rank.
  ///
  /// \param task the task
  /// \param trank the rank of the task
  ///
  /// \retunrs the average computation cost
  double computeCompCost(const kmp_target_task_t &Task, LegacyDeviceId Device);

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
  kmp_target_task_map_t *TaskMap;
  /// State of the scheduler
  LegacyScheduleMap &Schedule;
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
  /// Computation map
  ComputationMap ComputationCost;
  /// File where to dump HEFT information
  std::string HEFTLogPath;
  /// JSON object with profile info
  // json::Object ProfileInfo;
};

} // namespace detail

/// Implementation of the scheduler interface for HEFT.
///
/// This class provides a simple interface that external users can call to run
/// the HEFT scheduler.
///
/// \see LegacyHEFTContext
class LegacyHEFTScheduler final : public LegacyBaseScheduler {
public:
  /// Default constructor.
  LegacyHEFTScheduler() : Ctx(this->schedule_map, this->task_map) {}

  /// Destructor.
  ~LegacyHEFTScheduler() {}

public:
  std::unordered_map<int32_t, std::set<int>> dataTaskSuccessorRanks;

private:
  /// Run HEFT scheduling algorithm.
  void scheduleGraph() override;

  /// Clear context after acquiring new graph.
  void acquireNewGraphPost() override;

  /// Do nothing.
  void preScheduleTask(int32_t task_id, int32_t) override {}

public:
  /// Dump the task graph with HEFT-specific information.
  void dumpTaskGraph() override;

  std::set<int> getDataTaskSuccessors(int32_t task_id) override;

private:
  detail::LegacyHEFTContext Ctx;

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

#endif /* _OMPTARGET_LEGACY_HEFT_H */
