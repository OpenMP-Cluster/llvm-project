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

void RoundRobinScheduler::preScheduleTask(kmp_int32 task_id, kmp_int32 NumDevices) {

  assert((*get_schedule_map(tdg->tdg_id)).count(task_id) == 1);
  auto &schedule_data = (*get_schedule_map(tdg->tdg_id))[task_id];
  auto &task = schedule_data.task;

  if (schedule_data.rank != -1)
    return;

  schedule_data.state = TaskState::SCHEDULED;
  // Target data task are all mapped to the same device because we noticed
  // better performance this way (we cannot predict where the data is actually
  // going to be used with round-robin strategy).
  schedule_data.rank =
      isTargetDataTask(task)
          ? 1
          : (task_counter++ % NumDevices) + 1;
}
void RoundRobinScheduler::scheduleGraph() {
  const kmp_int32 num_devices = __kmpc_get_num_devices();

  for (auto &entry : *get_schedule_map(tdg->tdg_id)) {
    preScheduleTask(entry.first, num_devices);
  }

  addMemoryTasks();
}