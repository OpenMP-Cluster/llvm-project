//===---------------- scheduler.cpp - Target-Task scheduler ---------------===//
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

#include "scheduler.h"
#include "device.h"
#include "private.h"
#include "rtl.h"
#include "PluginManager.h"

// Scheduler interface in libomptarget
// =============================================================================

int Schedule(ident_t *Loc, int DeviceId) {

  DP("[SCHEDULER] Asked to schedule for Device%d (start)\n",DeviceId);

  const int32_t TaskId = __kmpc_target_task_current_task_id();
  DP("[SCHEDULER] Asked to schedule for Device %d task id : %d\n",DeviceId, TaskId);

  // Acquire the scheduled device id.
  int ScheduledDevId = __kmpc_scheduleTask(TaskId);
  DP("[SCHEDULER] Asked to schedule for Device %d task id : %d; scheduled Device: %d\n",DeviceId, TaskId, ScheduledDevId);

  if (TaskId != -1) {
    __kmpc_setTaskState(TaskId, TaskState::DISPATCHED);
  }

  DP("[SCHEDULER] Asked to schedule for Device%d (end)\n",DeviceId);
  return ScheduledDevId;
}

void PinTask(ident_t *Loc, int DeviceId) {
  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    __kmpc_setTaskState(TaskId, TaskState::DISPATCHED);
    __kmpc_pinTask(TaskId, DeviceId);
    DP("[SCHEDULER] Pinning task %d at device %d!\n", TaskId, DeviceId);
  }
}

void MarkExecutedTask() {
  // Non-mpi-target devices were not scheduled.
  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    __kmpc_setTaskState(TaskId, TaskState::EXECUTED);
  }
}

void MarkExecutedPinnedTask(const DeviceTy &Device) {
  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    // We must have the scheduler initialized up to this point.
    __kmpc_setTaskState(TaskId, TaskState::EXECUTED);
  }
}

int32_t GetCurrentTargetTaskID() {
  return __kmpc_target_task_current_task_id();
}

