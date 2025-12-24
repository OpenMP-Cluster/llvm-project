//===---------------- kmp_target_roundrobin.h - Target-Task scheduler
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the round-robin scheduler.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_roundrobin.h"

void RoundRobinScheduler::preScheduleTask(int32_t task_id, int32_t NumDevices) {

  assert(tdg->scheduling_info->schedule_map.count(task_id) == 1);
  auto &schedule_data = tdg->scheduling_info->schedule_map[task_id];
  auto &task = schedule_data.task;

  schedule_data.state = TaskState::SCHEDULED;
  // Target data task are all mapped to the same device because we noticed
  // better performance this way (we cannot predict where the data is actually
  // going to be used with round-robin strategy).
  schedule_data.rank =
      isTargetDataTask(task)
          ? 1
          : (task_counter++ % NumDevices) + 1;
}
