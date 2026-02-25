/*
 * kmp_target.cpp -- OpenMP integration with target library
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <list>
#include <functional>

#define assertm(exp, msg) assert(((void)msg, exp));

// Helper functions
// =============================================================================
template <typename T> static T *thread_malloc(kmp_info_t *thread, size_t size) {
  // Use fast user space allocation if an OpenMP thread context is available.
  if (thread != nullptr) {
#if USE_FAST_MEMORY
    return (T *)__kmp_fast_allocate(thread, size * sizeof(T));
#else
    return (T *)__kmp_thread_malloc(thread, size * sizeof(T));
#endif
  }

  // Fallback to normal allocation otherwise.
  return (T *)__kmp_allocate(size * sizeof(T));
}

// Returns the thread info of the current OpenMP thread. If the caller is not an
// OpenMP thread, returns nullptr.
static kmp_info_t *get_omp_thread() {
  // Get the current omp global thread id (valid if gtid >= 0).
  const int omp_gtid = __kmp_get_gtid();

  if (omp_gtid >= 0) {
    return __kmp_thread_from_gtid(omp_gtid);
  }

  return nullptr;
}

// Target task data
// =============================================================================
kmp_target_task_data::kmp_target_task_data(kmp_int64 device_id_p,
                                           void *outlined_fn_id_p,
                                           kmp_int32 num_args_p,
                                           void **args_base_p, void **args_p,
                                           kmp_int64 *arg_sizes_p,
                                           kmp_int64 *arg_types_p)
    : device_id(device_id_p), outlined_fn_id(outlined_fn_id_p),
      num_args(num_args_p) {
  allocBuffers();
  copyBuffers(args_base_p, args_p, arg_sizes_p, arg_types_p);
}

kmp_target_task_data::~kmp_target_task_data() { deleteBuffers(); }

void *kmp_target_task_data::operator new(size_t size) {
  // TODO: Check why we cannot use thread fast allocation without getting
  // address errors when freeing them.
  return __kmp_allocate(size);
}

void kmp_target_task_data::operator delete(void *ptr) { __kmp_free(ptr); }

kmp_target_task_data &
kmp_target_task_data::operator=(const kmp_target_task_data &other) {
  // Skip self assignment.
  if (this == &other) {
    return *this;
  }

  // Realloc internal arrays on size change.
  if (num_args != other.num_args) {
    deleteBuffers();
    num_args = other.num_args;
    allocBuffers();
  }

  // Assign and copy new data from other.
  device_id = other.device_id;
  outlined_fn_id = other.outlined_fn_id;
  copyBuffers(other.args_base, other.args, other.arg_sizes, other.arg_types);

  return *this;
}

void kmp_target_task_data::allocBuffers() {
  KG_TRACE(10,("kmp_target_task_data::allocBuffers (start)\n"));
  if (num_args > 0) {
    // TODO: Check why we cannot use thread fast allocation without getting
    // address errors when freeing them.
    args_base = (void **)__kmp_allocate(sizeof(void *) * num_args);
    args = (void **)__kmp_allocate(sizeof(void *) * num_args);
    arg_sizes = (kmp_int64 *)__kmp_allocate(sizeof(kmp_int64) * num_args);
    arg_types = (kmp_int64 *)__kmp_allocate(sizeof(kmp_int64) * num_args);
  }
  KG_TRACE(10,("kmp_target_task_data::allocBuffers (end)\n"));
}

void kmp_target_task_data::deleteBuffers() {
  KG_TRACE(10,("kmp_target_task_data::deleteBuffers (start)\n"));
  if (num_args > 0) {
    __kmp_free(args_base);
    __kmp_free(args);
    __kmp_free(arg_sizes);
    __kmp_free(arg_types);
  }
  args_base = nullptr;
  args = nullptr;
  arg_sizes = nullptr;
  arg_types = nullptr;
  KG_TRACE(10,("kmp_target_task_data::deleteBuffers (end)\n"));
}

void kmp_target_task_data::copyBuffers(void **args_base_p, void **args_p,
                                       kmp_int64 *arg_sizes_p,
                                       kmp_int64 *arg_types_p) {
  KG_TRACE(10,("kmp_target_task_data::copyBuffers (start)\n"));
  if (num_args > 0) {
    std::copy_n(args_base_p, num_args, args_base);
    std::copy_n(args_p, num_args, args);
    std::copy_n(arg_sizes_p, num_args, arg_sizes);
    std::copy_n(arg_types_p, num_args, arg_types);
  }
  KG_TRACE(10,("kmp_target_task_data::copyBuffers (end)\n"));
}

// Task and graph management
// =============================================================================

kmp_int32 __kmpc_target_task_current_task_id() {
  const kmp_info_t *thread = get_omp_thread();

  if (thread == nullptr) {
    return -1;
  }

  const kmp_taskdata_t *task = thread->th.th_current_task;

  // Only explicit tasks can be created by the user and should be accounted.
  // Implicit tasks are automatically generated by the OpenMP Runtime and should
  // not be considered in the target-task integration.
  if (task == nullptr || task->td_flags.tasktype != TASK_EXPLICIT) {
    return -1;
  }

  return thread->th.th_current_task->td_task_id;
}

//Add the array of the dependencies to a task
//TODO : This gestion of the dependencies can be simplified

void __kmpc_set_deps(kmp_node_info* task, kmp_int32 ndeps,
                    const kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                    const kmp_depend_info_t *dep_list_noalias){

  KG_TRACE(10,("__kmpc_set_deps (start)\n"));
  // Get general data
  kmp_int32 omp_gtid = __kmp_get_gtid();
  kmp_info_t *thread = __kmp_thread_from_gtid(omp_gtid);
  // Process dependencies
  task->deps = thread_malloc<kmp_target_dep_t>(thread, ndeps + ndeps_noalias);
  task->num_deps = 0;

  // Filter aliased addresses and add dependencies to task
  for (int i = 0; i < ndeps; ++i) {
    const kmp_depend_info_t &dep = dep_list[i];
    bool is_alias = false;

    for (int j = 0; j < task->num_deps; ++j) {
      kmp_target_dep_t &prev_dep = task->deps[j];
      if (prev_dep.addr == dep.base_addr) {
        prev_dep.dep_type.in |= dep.flags.in;
        prev_dep.dep_type.out |= dep.flags.out;
        prev_dep.dep_type.mtx |= dep.flags.mtx;
        is_alias = true;
        break;
      }
    }

    if (!is_alias) {
      task->deps[task->num_deps++] = kmp_target_dep_t{
          .addr = dep.base_addr,
          .dep_type = {.in = static_cast<bool>(dep.flags.in),
                       .out = static_cast<bool>(dep.flags.out),
                       .mtx = static_cast<bool>(dep.flags.mtx)}};
    }
  }

  // Add no-alias dependencies
  for (int i = 0; i < ndeps_noalias; ++i) {
    const kmp_depend_info_t &dep = dep_list_noalias[i];
    task->deps[task->num_deps + i] =
        kmp_target_dep_t{.addr = dep.base_addr,
                         .dep_type = {.in = static_cast<bool>(dep.flags.in),
                                      .out = static_cast<bool>(dep.flags.out),
                                      .mtx = static_cast<bool>(dep.flags.mtx)}};
  }
  task->num_deps += ndeps_noalias;

  KG_TRACE(10,("__kmpc_set_deps (end)\n"));
}

