//===---------------- kmp_target_roundrobin.h - Target-Task scheduler
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the round-robin scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_ROUNDROBIN_H
#define _OMPTARGET_ROUNDROBIN_H

#include "kmp_target_basescheduler.h"

#include <atomic>

/// Round-Robin scheduler
///
/// Maps tasks to devices in circular order.
class RoundRobinScheduler final : public BaseScheduler {
public:
  RoundRobinScheduler() = default;
  ~RoundRobinScheduler() { dumpTaskGraph(); }

private:
  /// Do nothing.
  void scheduleGraph() override {}

  /// Do nothing.
  void acquireNewGraphPost() override {}

  void resetWeight() override {}

  /// Schedules the task using round robin.
  void preScheduleTask(int32_t task_id, int32_t NumDevices) override;

private:
  /// Number of #preScheduleTask calls.
  std::atomic<size_t> task_counter;
};

#endif /* _OMPTARGET_ROUNDROBIN_H */
