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

// Scheduler implementation
// =============================================================================

//fixme: this function should be in libomp
//void DeinitScheduler() { __kmpc_DeinitScheduler(); }

int Schedule(ident_t *Loc, int DeviceId) {
  // Do not schedule non-mpi-target devices.
  const RTLInfoTy *DeviceRTL = PM->Devices[DeviceId]->RTL;
//  if (!DeviceRTL->IsMPI || DeviceRTL->DisableDataManager) {
//    return DeviceId;
//  }

//  DP("[SCHEDULER] Asked to schedule for %s[%d]\n",
//  DeviceRTL->RTLName.c_str(),
//     DeviceId);

  const int32_t TaskId = __kmpc_target_task_current_task_id();
//  static const bool MustWait = [] {
//    const char *str = std::getenv("OMPCLUSTER_BLOCKING_SCHEDULER");
//    if (str) {
//      return (bool)atoi(str);
//    }
//    return false;
//  }();

  // Acquire the scheduled device id.
  int ScheduledDevId = -1;
//  int OldExecCount = -1;
  std::atomic<int> *ExecCount = nullptr;
//  do {
    ScheduledDevId = DeviceRTL->Idx + __kmpc_scheduleTask(TaskId);
//    ExecCount = &PM->Devices[ScheduledDevId]->ExecCount;

    // Set OldExecCount to its expected value for a successfull scheduling.
//    OldExecCount = 0;
//  } while (MustWait && !ExecCount->compare_exchange_weak(OldExecCount, 1));
  // Update exec count when scheduler is configured to not be blocking.
//  if (!MustWait) {
    (*ExecCount)++;
//  }
  assert(*ExecCount > 0);

  if (TaskId != -1) {
    __kmpc_setTaskState(TaskId, TaskState::DISPATCHED);
  }
  //DP("[SCHEDULER] Using %s[%d]!\n", DeviceRTL->RTLName.c_str(), ScheduledDevId);
  //__kmpc_dumpTaskGraph();
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

void MarkExecutedTask(int DeviceId) {
  // Non-mpi-target devices were not scheduled.
  DeviceTy &Device = *PM->Devices[DeviceId];
  //if (!Device.RTL->IsMPI || Device.RTL->DisableDataManager) {
  //  return;
  //}

  // Update counter of in flight regions.
  Device.ExecCount--;
  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    __kmpc_setTaskStateWithDevice(TaskId, DeviceId, TaskState::EXECUTED);
  }
}

//void MarkFailedTask(int32_t task_id, const DeviceTy &Device) {
  // Non-mpi-target devices were not scheduled.
//  if (!Device.RTL->IsMPI || Device.RTL->DisableDataManager) {
//    return;
//  }
//
//  __kmpc_setTaskState(task_id, TaskState::FAILED);
//}

void MarkExecutedPinnedTask(const DeviceTy &Device) {
  // Non-mpi-target devices were not scheduled.
  //if (!Device.RTL->IsMPI || Device.RTL->DisableDataManager) {
  //  return;
  //}

  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    // We must have the scheduler initialized up to this point.
    __kmpc_setTaskState(TaskId, TaskState::EXECUTED);
  }
}

//bool HasOutDep(int32_t task_id, int64_t addr) {
//  return __kmpc_getDep(task_id, addr).dep_type.out;
//}

int32_t GetCurrentTargetTaskID() {
  return __kmpc_target_task_current_task_id();
}

