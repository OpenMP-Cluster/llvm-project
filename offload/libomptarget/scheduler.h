//===----------------- scheduler.h - Target-Task scheduler ----------------===//
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

#ifndef _OMPTARGET_SCHEDULER_H
#define _OMPTARGET_SCHEDULER_H

#include <cstdint>
#include <set>

#include <Shared/SourceInfo.h>

struct RTLInfoTy;
struct DeviceTy;

// Schedule a target region among the many devices of the same type as DeviceID.
// Currently only used by the MPI plugin.
int Schedule(ident_t *Loc, int DeviceId);

/// Pins the current task at a specific device.
///
/// \note this function does not exclusively allocate the device.
void PinTask(ident_t *Loc, int DeviceId);

// Mark the target task as executed and free the allocated device.
void MarkExecutedTask();

/// Marks a pinned task as executed.
///
/// \note since pinned tasks does not allocate devices, no free is needed.
void MarkExecutedPinnedTask(const DeviceTy &Device);

int32_t GetCurrentTargetTaskID();

/// Get the outlined function ID of the current task.
const void *GetCurrentOutlinedFnID();



#ifdef __cplusplus
extern "C" {
#endif

void __kmpc_dmdas_target_dispatched(int32_t task_id)
    __attribute__((weak));

int32_t __kmpc_dmdas_get_task_device(int32_t task_id)
    __attribute__((weak));

enum class TaskState {
  UNDEFINED = -1,
  CREATED,
  DEFERRED, // Only used by HEFTScheduler
  SCHEDULED,
  DISPATCHED,
  EXECUTED,
  FAILED,
  SAVED
};

struct kmp_target_dep {
  int64_t addr;
  struct {
    bool in : 1;
    bool out : 1;
  } dep_type;
};

int32_t __kmpc_target_task_current_task_id() __attribute__((weak));
void __kmpc_DeinitScheduler() __attribute__((weak));
int32_t __kmpc_scheduleTask(int32_t task_id) __attribute__((weak));
void __kmpc_pinTask(int32_t task_id, int DeviceId) __attribute__((weak));
void __kmpc_setTaskState(int32_t task_id, TaskState new_state)
    __attribute__((weak));
void __kmpc_setTaskStateWithDevice(int32_t task_id, int device_id,
                                   TaskState new_state) __attribute__((weak));
kmp_target_dep __kmpc_getDep(int32_t task_id, int64_t addr)
    __attribute__((weak));
void __kmpc_dumpTaskGraph() __attribute__((weak));
const void *__kmpc_GetCurrentOutlinedFnID() __attribute__((weak));
#ifdef __cplusplus
} // extern "C"
#endif

#endif // _OMPTARGET_SCHEDULER_H
