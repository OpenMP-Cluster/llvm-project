//===------------- kmp_target_graph_roundrobin.h - Target-Task scheduler
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the graph based round-robin scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_GRAPH_ROUNDROBIN_H
#define _OMPTARGET_GRAPH_ROUNDROBIN_H

#include "kmp_target_basescheduler.h"

/// Graph Round-Robin scheduler
///
/// Statically maps tasks to devices in circular order using the task graph.
class GraphRoundRobinScheduler final : public BaseScheduler {
public:
  GraphRoundRobinScheduler() = default;
  ~GraphRoundRobinScheduler() { dumpTaskGraph(); }

private:
  /// Run scheduling algorithm.
  void scheduleGraph() override;

  /// Do nothing.
  void acquireNewGraphPost() override {}
  
  void resetWeight() override {}

  /// Do nothing.
  void preScheduleTask(kmp_int32 task_id, kmp_int32 NumDevices) override {}
};

#endif /* _OMPTARGET_GRAPH_ROUNDROBIN_H */
