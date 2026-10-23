#ifndef KMP_DMDAS_H
#define KMP_DMDAS_H

#include "kmp.h"

bool __kmp_use_dmdas_runtime();

kmp_int32 __kmp_dmdas_task_with_deps(
    ident_t *loc_ref,
    kmp_int32 gtid,
    kmp_task_t *new_task,
    kmp_int32 ndeps,
    kmp_depend_info_t *dep_list,
    kmp_int32 ndeps_noalias,
    kmp_depend_info_t *noalias_dep_list);


void __kmp_dmdas_start_scheduler();

void __kmp_dmdas_notify_task_completed(kmp_task_t *task);

void __kmp_dmdas_notify_task_started(kmp_task_t *task);

void __kmp_dmdas_notify_target_completed(int32_t task_id);

void __kmp_dmdas_native_ready(kmp_task_t *task);

void __kmp_dmdas_run_ready_tasks();

bool __kmp_dmdas_owns_task(kmp_task_t *task);

bool __kmp_dmdas_all_tasks_completed();

void __kmp_dmdas_prepare_replay(kmp_taskdata_t *parent_task);

void __kmp_dmdas_reset_for_replay();

extern "C"
void __kmpc_dmdas_save_perf_model();

extern void __kmp_dmdas_invoke_task(
    kmp_int32 gtid,
    kmp_task_t *task);

enum kmp_dmdas_perf_feature_kind {
    KMP_DMDAS_FEATURE_DATA_BYTES = 0,
    KMP_DMDAS_FEATURE_ITERATIONS = 1,
    KMP_DMDAS_FEATURE_FLOPS = 2,
    KMP_DMDAS_FEATURE_USER = 100
};

void __kmpc_dmdas_add_current_task_perf_feature(
    int kind,
    double value);

extern "C"
void __kmpc_dmdas_add_next_task_perf_feature(
    int kind,
    double value);

#endif