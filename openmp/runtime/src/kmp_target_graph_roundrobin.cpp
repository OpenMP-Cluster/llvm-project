//===------------ kmp_target_graph_roundrobin.cpp - Target-Task scheduler
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the graph based round-robin scheduler.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_graph_roundrobin.h"
#include <algorithm>
#include "kmp.h"

void GraphRoundRobinScheduler::scheduleGraph() {

  // Graph based round-robin algorithm.
  int task_counter = 0;

  int NumDevices = __kmpc_get_num_devices();
  for (auto &map_entry : tdg->scheduling_info->schedule_map) {
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    // Do not reschedule already executed or saved tasks
    if (task_schedule.state > TaskState::SCHEDULED &&
        task_schedule.state != TaskState::FAILED)
      continue;

    assert(task_schedule.task->is_target_task);
    assert(task_schedule.state <= TaskState::SCHEDULED);

    task_schedule.state = TaskState::SCHEDULED;
    if (isTargetDataTask(task)) {
      // FIXME: Map target data task to the proxy device, this resolve a problem
      // with data manager but should not be necessary. It would actually be
      // better to map them using the same strategy as the HEFT.
      task_schedule.rank = 1;
    } else {
      task_schedule.rank = (task_counter++ % NumDevices) + 1;
    }
  }
  
  addMemoryTasks();

  assert(std::none_of(tdg->scheduling_info->schedule_map.begin(), tdg->scheduling_info->schedule_map.end(),
                      [](const auto &entry) {
                        return entry.second.state < TaskState::SCHEDULED;
                      }));
}
