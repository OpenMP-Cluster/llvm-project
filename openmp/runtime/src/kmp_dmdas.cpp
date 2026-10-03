/*
 * kmp_dmdas.cpp
 */

#include "kmp.h"
#include <queue>
#include <condition_variable>
#include "kmp_dmdas.h"

#include <vector>
#include <unordered_map>
#include <deque>
#include <thread>
#include <chrono>
#include <cfloat>
#include <mutex>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <dlfcn.h>



struct RuntimeWorker;
struct RuntimeNode;
struct RuntimeData;
struct RuntimeTransfer;

enum class PerfFeatureKind {
    DATA_BYTES,
    ITERATIONS,
    FLOPS,
    USER
};

struct PerfFeature {
    PerfFeatureKind kind;
    double value;
};

struct PerfFeatures {
    std::vector<PerfFeature> values;

    void clear()
    {
        values.clear();
    }

    void add(
        PerfFeatureKind kind,
        double value)
    {
        values.push_back({kind, value});
    }

    bool empty() const
    {
        return values.empty();
    }

    double first_value() const
    {
        if (values.empty())
            return 0.0;

        return values[0].value;
    }
};

struct DMDASTask {

    int instance_id; // Unique identifier for this task instance in the current task graph.

    uintptr_t model_id = 0; // Identifier of the performance model associated with this task.

    PerfFeatures perf_features; // Performance features associated with this task.

    kmp_int32 gtid;     // Global thread ID of the thread that created the task.

    kmp_task_t *omp_task; // OpenMP task associated with this runtime task.

    kmp_taskdata_t *current_task;    // Internal OpenMP task metadata.

    std::vector<kmp_depend_info_t> deps;    // OpenMP dependency descriptors associated with the task.

    std::vector<size_t> dep_sizes;    // Size in bytes of each dependency.

    std::vector<DMDASTask*> predecessors;    // Tasks that must complete before this task becomes ready.

    std::vector<DMDASTask *> successors;    // Tasks that depend on the completion of this task.

    std::atomic<int> remaining_dependencies;    // Number of predecessors that have not completed yet.

    bool submitted = false;    // True once the task has been submitted to a DMDAS worker queue.

    bool completed = false;    // True once the task execution has completed.

    RuntimeWorker *assigned_worker = nullptr;    // Worker/device selected by DMDAS to execute this task.

    std::vector<RuntimeTransfer*> transfers;    // Data transfers required before executing the task.

    /*
     * Async-transfer state.
     *
     * These fields are not used by the scheduler yet. They only describe
     * whether the data transfers associated with this task have been started
     * and whether all of them have completed.
     */
    std::atomic<int> pending_transfers{0};
    std::atomic<bool> transfers_started{false};
    std::atomic<bool> transfers_ready{false};

    double priority = 0.0;    // Scheduling priority of the task.

    double bottom_level = 0.0;    // Bottom-level value used to prioritize tasks on the critical path.

    bool native_ready = false;     // True when the task is ready according to the native OpenMP depedency mechanism.

    std::chrono::high_resolution_clock::time_point submit_time ;// Time at which DMDAS submits the task to the OpenMP runtime.

    std::chrono::high_resolution_clock::time_point execution_start; // Time at which OpenMP actually starts executing the task.

    std::chrono::high_resolution_clock::time_point target_start; // Time at which the target region of the task starts executing on the device.

    bool submitted_to_openmp = false; // True once the task has been submitted to the OpenMP runtime for execution.

    bool execution_started = false; // True once the OpenMP runtime has started executing the task.

    bool target_started = false; // True once the target region of the task has started executing on the device.


};

struct RuntimeNode
{
    int id;

    std::vector<RuntimeWorker*> workers;

    /*
     * Directional communication model.
     *
     * For a source node S and destination node D:
     *
     *   transfer_time =
     *       S.latency[D.id] +
     *       bytes / S.bandwidth[D.id]
     */
    std::vector<double> bandwidth;

    std::vector<double> latency;
};

static double transfer_time(
    RuntimeNode *src,
    RuntimeNode *dst,
    size_t bytes);

struct RuntimeWorker
{
    int id;

    int device_id;

    // Raw OMPC/libomptarget device type:
    // 0 = UNKNOWN, 1 = CPU, 2 = CUDA, 3 = HSA.
    int device_type;

    RuntimeNode *node;

    bool busy;

    DMDASTask *current_task;

    double available_time = 0.0;

    std::deque<DMDASTask*> ready_queue;
};

struct RuntimeCopy
{
    RuntimeNode *node;

     enum State
    {
        INVALID,
        TRANSFERRING,
        VALID
    };

    State state = INVALID;
};

struct RuntimeData
{
    void *addr;

    size_t size;

    std::vector<RuntimeCopy> copies;

};

struct RuntimeTransfer
{
    RuntimeData *data = nullptr;
    RuntimeNode *src = nullptr;
    RuntimeNode *dst = nullptr;

    double completion_time;

    std::atomic<bool> finished{false};

    RuntimeWorker *dma_worker = nullptr;

    bool submitted = false;

    bool started = false;

    enum RuntimeTransferState
    {
        TRANSFER_PENDING,
        TRANSFER_RUNNING,
        TRANSFER_FINISHED
    };

    std::atomic<RuntimeTransferState> state{
        TRANSFER_PENDING
    };
};

class RuntimeDataManager
{
public:

    RuntimeData &get(void *addr, size_t size);

    bool has_valid_copy(RuntimeData &, RuntimeNode *);

    RuntimeNode *source(RuntimeData &);

    double transfer_cost(RuntimeData &, RuntimeNode *);

    void read(RuntimeData &, RuntimeNode *);

    void write(RuntimeData &, RuntimeNode *);

    void reset_copies();



private:

    std::unordered_map<void *, RuntimeData> data_map;
};



class RuntimeScheduler
{
public:

    void submit(DMDASTask *);

    DMDASTask *pick_next(RuntimeWorker *);

    void task_completed(DMDASTask *);

    void recompute_priorities();

private:

    void insert_sorted(RuntimeWorker *, DMDASTask *);

    double compute_bottom_level(DMDASTask *,
                                std::unordered_map<DMDASTask*, bool> &);

    void update_priorities();
};

class RuntimeExecutor
{
public:
    void execute(RuntimeWorker *, DMDASTask *);
};

struct TransferKey
{
    RuntimeData *data;
    RuntimeNode *src;
    RuntimeNode *dst;

    bool operator==(const TransferKey &other) const
    {
        return data == other.data &&
               src == other.src &&
               dst == other.dst;
    }
};

struct TransferKeyHash
{
    size_t operator()(const TransferKey &key) const
    {
        size_t h1 = std::hash<void *>{}(key.data);
        size_t h2 = std::hash<void *>{}(key.src);
        size_t h3 = std::hash<void *>{}(key.dst);

        size_t seed = h1;

        seed ^= h2 + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);

        seed ^= h3 + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);

        return seed;
    }
};

class RuntimeTransferManager
{
public:

    RuntimeTransfer *submit(RuntimeData *, RuntimeNode *, RuntimeNode *);

    void progress(RuntimeTransfer *);

    bool completed(RuntimeTransfer *);

    RuntimeTransfer *find(RuntimeData *, RuntimeNode *, RuntimeNode *);

    void clear_finished();

private:

    std::vector<RuntimeTransfer*> transfers;

    std::unordered_map<
        TransferKey,
        RuntimeTransfer *,
        TransferKeyHash
    > active_transfers;

    std::mutex transfers_mutex;
};


/*
 * Persistent transfer executor for DMDAS.
 *
 * Transfer threads are created once and reused to execute asynchronous
 * data transfers without blocking the scheduler.
 */
class RuntimeTransferExecutor
{
public:
    RuntimeTransferExecutor() = default;

    ~RuntimeTransferExecutor()
    {
        stop();
    }

    void start(size_t nthreads)
    {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (!threads.empty())
            return;

        stopping = false;

        for (size_t i = 0; i < nthreads; ++i) {
            threads.emplace_back(
                [this]() {
                    worker_loop();
                });
        }
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);

            if (threads.empty())
                return;

            stopping = true;
        }

        queue_cv.notify_all();

        for (std::thread &thread : threads) {
            if (thread.joinable())
                thread.join();
        }

        threads.clear();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);

            while (!queue.empty())
                queue.pop();

            stopping = false;
        }
    }

    void submit(DMDASTask *task)
    {
        if (task == nullptr)
            return;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            queue.push(task);
        }

        queue_cv.notify_one();
    }

private:
    void worker_loop();

    std::queue<DMDASTask *> queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<std::thread> threads;
    bool stopping = false;
};



struct MultiRegressionModel
{
    /*
     * Number of explanatory variables.
     *
     * Examples:
     *   ITERATIONS -> feature_count = 1
     *   M, N, K    -> feature_count = 3
     *
     * The beta_0 intercept is added automatically, so the system has
     * feature_count + 1 coefficients.
     */
    size_t feature_count = 0;

    /*
     * Number of observed samples.
     */
    int n = 0;

    /*
     * Sufficient statistics for linear regression:
     *
     *     y = beta_0 + beta_1*x1 + ... + beta_k*xk
     *
     * Accumulate:
     *
     *     XTX = X^T X
     *     XTy = X^T y
     *
     * This allows the model to be reconstructed without retaining
     * all samples in memory.
     */
    std::vector<double> xtx;
    std::vector<double> xty;

    /*
     * Computed coefficients.
     *
     * beta[0] = intercept
     * beta[1] = coefficient for feature 0
     * ...
     */
    std::vector<double> beta;

    /*
     * Sum of observed execution times, used as a fallback until enough
     * samples are available to determine the model.
     */
    double sy = 0.0;

    void initialize(size_t count)
    {
        if (feature_count != 0)
            return;

        feature_count = count;

        size_t dim = feature_count + 1;

        xtx.assign(dim * dim, 0.0);
        xty.assign(dim, 0.0);
        beta.assign(dim, 0.0);
    }

    /*
     * Solve the small linear system
     *
     *     A beta = b
     *
     * using Gaussian elimination with partial pivoting. Performance
     * models have few features, so a simple dense implementation suffices.
     */
    static bool solve_linear_system(
        std::vector<double> A,
        std::vector<double> b,
        std::vector<double> &solution)
    {
        const size_t dim = b.size();

        solution.assign(dim, 0.0);

        for (size_t col = 0; col < dim; ++col) {

            size_t pivot = col;

            for (size_t row = col + 1; row < dim; ++row) {
                if (fabs(A[row * dim + col]) >
                    fabs(A[pivot * dim + col])) {
                    pivot = row;
                }
            }

            if (fabs(A[pivot * dim + col]) < 1e-12)
                return false;

            if (pivot != col) {
                for (size_t j = 0; j < dim; ++j) {
                    std::swap(
                        A[col * dim + j],
                        A[pivot * dim + j]);
                }

                std::swap(b[col], b[pivot]);
            }

            double divisor = A[col * dim + col];

            for (size_t j = col; j < dim; ++j)
                A[col * dim + j] /= divisor;

            b[col] /= divisor;

            for (size_t row = 0; row < dim; ++row) {

                if (row == col)
                    continue;

                double factor =
                    A[row * dim + col];

                for (size_t j = col; j < dim; ++j) {
                    A[row * dim + j] -=
                        factor * A[col * dim + j];
                }

                b[row] -= factor * b[col];
            }
        }

        solution = std::move(b);

        return true;
    }

    void update(
        const std::vector<double> &features,
        double y)
    {
        if (features.empty())
            return;

        initialize(features.size());

        /*
         * A model must always use the same number of features.
         */
        if (features.size() != feature_count)
            return;

        const size_t dim = feature_count + 1;

        /*
         * x = [1, f1, f2, ..., fk], where 1 represents the intercept.
         */
        std::vector<double> x(dim, 1.0);

        for (size_t i = 0; i < feature_count; ++i)
            x[i + 1] = features[i];

        n++;
        sy += y;

        for (size_t i = 0; i < dim; ++i) {

            xty[i] += x[i] * y;

            for (size_t j = 0; j < dim; ++j) {
                xtx[i * dim + j] +=
                    x[i] * x[j];
            }
        }

        /*
         * Il faut au minimum autant d'observations
         * que de coefficients.
         */
        if (n < static_cast<int>(dim))
            return;

        std::vector<double> solution;

        if (solve_linear_system(
                xtx,
                xty,
                solution)) {
            beta = std::move(solution);
        }
    }

    double predict(
        const std::vector<double> &features) const
    {
        if (n == 0)
            return 0.004;

        if (features.size() != feature_count)
            return sy / n;

        const size_t dim = feature_count + 1;

        /*
         * Tant que la régression n'est pas identifiable,
         * utiliser la moyenne observée.
         */
        if (n < static_cast<int>(dim))
            return sy / n;

        /*
         * beta peut encore être nul si X^T X était singulier
         * (par exemple toutes les observations identiques).
         */
        bool has_model = false;

        for (double value : beta) {
            if (fabs(value) > 1e-15) {
                has_model = true;
                break;
            }
        }

        if (!has_model)
            return sy / n;

        double prediction = beta[0];

        for (size_t i = 0; i < feature_count; ++i) {
            prediction +=
                beta[i + 1] * features[i];
        }

        return std::max(prediction, 0.0);
    }
};




static size_t get_feature_signature(
    const PerfFeatures &features)
{
    size_t seed = 0;

    for (const auto &feature : features.values) {

        size_t h =
            std::hash<int>{}(
                static_cast<int>(feature.kind));

        seed ^=
            h +
            0x9e3779b9 +
            (seed << 6) +
            (seed >> 2);
    }

    return seed;
}

struct PerfModelKey {

    uintptr_t model_id;
    int worker_id;

    size_t feature_signature;
    size_t feature_count;

    bool operator==(
        const PerfModelKey &other) const
    {
        return
            model_id ==
                other.model_id &&
            worker_id ==
                other.worker_id &&
            feature_signature ==
                other.feature_signature &&
            feature_count ==
                other.feature_count;
    }
};

struct PerfModelKeyHash {

    size_t operator()(const PerfModelKey &key) const
    {
        size_t h1 =
            std::hash<uintptr_t>{}(key.model_id);

        size_t h2 =
            std::hash<int>{}(key.worker_id);

        size_t h3 =
            std::hash<size_t>{}(
                key.feature_signature);

        size_t h4 =
            std::hash<size_t>{}(
                key.feature_count);

        size_t seed = h1;

        seed ^=
            h2 +
            0x9e3779b9 +
            (seed << 6) +
            (seed >> 2);

        seed ^=
            h3 +
            0x9e3779b9 +
            (seed << 6) +
            (seed >> 2);

        seed ^=
            h4 +
            0x9e3779b9 +
            (seed << 6) +
            (seed >> 2);

        return seed;
    }
};

class PerformanceModel
{
public:

    double compute_cost(
            DMDASTask *,
            RuntimeWorker *);

    double transfer_cost(
            RuntimeData *,
            RuntimeWorker *);

    void record_execution(
            DMDASTask *,
            RuntimeWorker *,
            double duration);

    int load(const char *path);
    void save(const char *path);
    void save_sample(
        const char *path,
        DMDASTask *task,
        RuntimeWorker *worker,
        double duration);

private:

    std::unordered_map<
        PerfModelKey,
        MultiRegressionModel,
        PerfModelKeyHash
    > models;
};

// Global variable

static std::vector<DMDASTask *> g_dmdas_tasks;

static std::unordered_map<void *, DMDASTask *> g_last_writer;
static std::unordered_map<void *, std::vector<DMDASTask *>> g_last_readers;

static std::unordered_map<kmp_task_t *, DMDASTask *> g_task_map;

static std::unordered_map<int32_t, DMDASTask *> g_task_id_map;

std::atomic<bool> scheduler_running(false);
std::atomic<bool> scheduler_pending(false);

static std::mutex dmdas_mutex;

static std::vector<RuntimeWorker> g_workers;

static RuntimeDataManager g_data_manager;
static RuntimeScheduler g_scheduler;
static RuntimeExecutor g_executor;
static PerformanceModel g_perf_model;

static std::vector<RuntimeNode> g_nodes;

static RuntimeTransferManager g_transfer_manager;

/*
 * Declared after g_transfer_manager so that it is destroyed first.
 * Its destructor stops and joins all persistent transfer threads.
 */
static RuntimeTransferExecutor g_transfer_executor;

static std::unordered_map<int, PerfFeatures>
    g_pending_perf_features;


static std::atomic<bool> g_perf_model_frozen{false};

static std::vector<size_t> g_compute_placement;


static size_t get_target_arg_size(
    DMDASTask *task,
    const kmp_depend_info_t &dep);


void RuntimeDataManager::reset_copies()
{
    for (auto &entry : data_map) {
        entry.second.copies.clear();
    }
}

static uintptr_t get_stable_model_id(void *outlined_fn_id)
{
    if (outlined_fn_id == nullptr)
        return 0;

    Dl_info info{};

    if (dladdr(outlined_fn_id, &info) == 0 ||
        info.dli_fbase == nullptr) {
        return reinterpret_cast<uintptr_t>(outlined_fn_id);
    }

    uintptr_t fn =
        reinterpret_cast<uintptr_t>(outlined_fn_id);

    uintptr_t base =
        reinterpret_cast<uintptr_t>(info.dli_fbase);

    uintptr_t offset = fn - base;


    return offset;
}

static std::string get_perf_model_path()
{
    const char *dir =
        getenv("OMPCLUSTER_PERF_MODEL_DIR");

    if (dir == nullptr || dir[0] == '\0')
        return "";

    return std::string(dir) + "/perf_models.dat";
}

static std::string get_perf_sample_path()
{
    const char *enabled =
        getenv("OMPCLUSTER_PERF_SAMPLES");

    if (enabled == nullptr ||
        enabled[0] == '\0' ||
        strcmp(enabled, "1") != 0)
        return "";

    const char *dir =
        getenv("OMPCLUSTER_PERF_MODEL_DIR");

    if (dir == nullptr || dir[0] == '\0')
        return "";

    return std::string(dir) + "/perf_samples.dat";
}

void __kmpc_dmdas_target_dispatched(int32_t task_id)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_id_map.find(task_id);

    if (it == g_task_id_map.end())
        return;

    DMDASTask *task = it->second;

    task->target_start =
        std::chrono::high_resolution_clock::now();

    task->target_started = true;
}

int32_t __kmpc_dmdas_get_task_device(int32_t task_id)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_id_map.find(task_id);

    if (it == g_task_id_map.end())
        return -1;

    DMDASTask *task = it->second;

    if (task == nullptr ||
        task->assigned_worker == nullptr)
        return -1;

    return task->assigned_worker->device_id;
}


static std::vector<double>
get_feature_values(
    const PerfFeatures &features)
{
    std::vector<double> values;

    values.reserve(
        features.values.size());

    for (const auto &feature :
         features.values) {
        values.push_back(
            feature.value);
    }

    return values;
}

void PerformanceModel::save(const char *path)
{
    if (path == nullptr || path[0] == '\0')
        return;

    std::ofstream file(path);

    if (!file.is_open()) {
        printf(
            "PERF MODEL: cannot save file %s\n",
            path);
        return;
    }

    file << "# OMPC DMDAS multidimensional performance models\n";
    file << "# format:\n";
    file << "# model_id worker_id feature_signature "
            "feature_count n sy "
            "xtx_count [xtx...] "
            "xty_count [xty...] "
            "beta_count [beta...]\n";

    for (const auto &entry : models) {

        const PerfModelKey &key =
            entry.first;

        const MultiRegressionModel &model =
            entry.second;

        file
            << std::hex
            << key.model_id
            << std::dec << " "
            << key.worker_id << " "
            << key.feature_signature << " "
            << key.feature_count << " "
            << model.n << " "
            << model.sy << " ";

        /*
         * X^T X
         */
        file << model.xtx.size();

        for (double value : model.xtx)
            file << " " << value;

        /*
         * X^T y
         */
        file << " " << model.xty.size();

        for (double value : model.xty)
            file << " " << value;

        /*
         * Regression coefficients.
         */
        file << " " << model.beta.size();

        for (double value : model.beta)
            file << " " << value;

        file << "\n";
    }


}

void PerformanceModel::save_sample(
        const char *path,
        DMDASTask *task,
        RuntimeWorker *worker,
        double duration)
{
    if (path == nullptr || path[0] == '\0')
        return;

    std::ofstream file(
        path,
        std::ios::out | std::ios::app);

    if (!file.is_open()) {
        printf(
            "PERF SAMPLE FILE: cannot open %s\n",
            path);
        return;
    }

    /*
     * Format:
     *
     * model_id worker_id feature_count
     *   kind_0 value_0
     *   kind_1 value_1
     *   ...
     * duration
     *
     * Stored on a single line.
     *
     * Example:
     *
     * 11038 0 1 1 100000 0.000373
     *
     * means:
     *   model      = 0x11038
     *   worker     = 0
     *   nfeatures  = 1
     *   kind       = ITERATIONS (1)
     *   value      = 100000
     *   duration   = 373 us
     */
    file
        << std::hex << task->model_id
        << std::dec << " "
        << worker->id << " "
        << task->perf_features.values.size();

    for (const auto &feature :
         task->perf_features.values) {

        file
            << " "
            << static_cast<int>(feature.kind)
            << " "
            << feature.value;
    }

    file
        << " "
        << duration
        << "\n";
}

int PerformanceModel::load(const char *path)
{
    if (path == nullptr || path[0] == '\0')
        return 0;

    std::ifstream file(path);

    if (!file.is_open()) {
        printf(
            "PERF MODEL: no existing model file %s\n",
            path);
        return 0;
    }

    std::string line;
    int loaded = 0;

    while (std::getline(file, line)) {

        if (line.empty() ||
            line[0] == '#')
            continue;

        std::istringstream iss(line);

        PerfModelKey key{};
        MultiRegressionModel model{};

        size_t xtx_count = 0;
        size_t xty_count = 0;
        size_t beta_count = 0;

        iss
            >> std::hex
            >> key.model_id
            >> std::dec
            >> key.worker_id
            >> key.feature_signature
            >> key.feature_count
            >> model.n
            >> model.sy
            >> xtx_count;

        model.feature_count =
            key.feature_count;

        if (iss.fail())
            continue;

        model.xtx.resize(
            xtx_count);

        for (size_t i = 0;
             i < xtx_count;
             ++i) {

            iss >> model.xtx[i];
        }

        iss >> xty_count;

        if (iss.fail())
            continue;

        model.xty.resize(
            xty_count);

        for (size_t i = 0;
             i < xty_count;
             ++i) {

            iss >> model.xty[i];
        }

        iss >> beta_count;

        if (iss.fail())
            continue;

        model.beta.resize(
            beta_count);

        for (size_t i = 0;
             i < beta_count;
             ++i) {

            iss >> model.beta[i];
        }

        if (iss.fail())
            continue;

        /*
         * Basic consistency check.
         *
         * For k features:
         *
         *   dim = k + 1
         *   XTX = dim x dim
         *   XTy = dim
         *   beta = dim
         */
        size_t dim =
            model.feature_count + 1;

        if (model.xtx.size() !=
                dim * dim ||
            model.xty.size() !=
                dim ||
            model.beta.size() !=
                dim) {

            printf(
                "PERF MODEL: invalid entry "
                "model=0x%lx worker=%d\n",
                static_cast<unsigned long>(
                    key.model_id),
                key.worker_id);

            continue;
        }

        models[key] =
            std::move(model);

        loaded++;
    }

    return loaded;
}

void __kmpc_dmdas_add_current_task_perf_feature(
    int kind,
    double value)
{
    int gtid = __kmp_get_gtid();

    if (gtid < 0)
        return;

    kmp_info_t *thread =
        __kmp_threads[gtid];

    if (!thread)
        return;

    kmp_taskdata_t *taskdata =
        thread->th.th_current_task;

    if (!taskdata)
        return;

    kmp_task_t *omp_task =
        KMP_TASKDATA_TO_TASK(taskdata);

    std::lock_guard<std::mutex> lock(
        dmdas_mutex);

    auto it =
        g_task_map.find(omp_task);

    if (it == g_task_map.end())
        return;

    DMDASTask *task =
        it->second;

    task->perf_features.add(
        static_cast<PerfFeatureKind>(kind),
        value);

}

void __kmpc_dmdas_add_next_task_perf_feature(
    int kind,
    double value)
{
    int gtid = __kmp_get_gtid();

    if (gtid < 0)
        return;

    std::lock_guard<std::mutex> lock(dmdas_mutex);

    g_pending_perf_features[gtid].add(
        static_cast<PerfFeatureKind>(kind),
        value);

}

bool __kmp_dmdas_owns_task(kmp_task_t *task) {


    std::lock_guard<std::mutex> lock(dmdas_mutex);
    return g_task_map.find(task) != g_task_map.end();
}

bool __kmp_dmdas_all_tasks_completed()
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    for (DMDASTask *task : g_dmdas_tasks) {
        if (!task->completed) {
            fprintf(stderr,
                    "[DMDAS-STUCK] "
                    "instance=%d omp_id=%d "
                    "completed=%d submitted=%d native_ready=%d "
                    "remaining=%d transfers=%zu pending=%d ready=%d "
                    "assigned=%p\\n",
                    task->instance_id,
                    KMP_TASK_TO_TASKDATA(task->omp_task)->td_task_id,
                    static_cast<int>(task->completed),
                    static_cast<int>(task->submitted),
                    static_cast<int>(task->native_ready),
                    task->remaining_dependencies.load(),
                    task->transfers.size(),
                    task->pending_transfers.load(),
                    static_cast<int>(task->transfers_ready.load()),
                    static_cast<void *>(task->assigned_worker));
            fflush(stderr);
            return false;
        }
    }

    return true;
}

void __kmp_dmdas_prepare_replay(kmp_taskdata_t *parent_task)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    for (DMDASTask *task : g_dmdas_tasks) {

        kmp_taskdata_t *td =
            KMP_TASK_TO_TASKDATA(task->omp_task);

        td->td_parent = parent_task;

        kmp_taskgroup_t *parent_taskgroup =
            parent_task->td_taskgroup;

        KMP_ATOMIC_INC(
            &parent_task->td_incomplete_child_tasks);

        if (parent_taskgroup) {
            KMP_ATOMIC_INC(&parent_taskgroup->count);
            td->td_taskgroup = parent_taskgroup;
        } else {
            td->td_taskgroup = nullptr;
        }

        if (parent_task->td_flags.tasktype == TASK_EXPLICIT) {
            KMP_ATOMIC_INC(
                &parent_task->td_allocated_child_tasks);
        }

        if (td->td_flags.hidden_helper) {
            td->td_flags.task_serial = FALSE;

            KMP_ATOMIC_INC(
                &__kmp_unexecuted_hidden_helper_tasks);
        }
    }
}

void RuntimeTransferExecutor::worker_loop()
{
    while (true) {
        DMDASTask *task = nullptr;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            queue_cv.wait(
                lock,
                [this]() {
                    return stopping || !queue.empty();
                });

            if (stopping && queue.empty())
                return;

            task = queue.front();
            queue.pop();
        }

        if (task == nullptr)
            continue;

        for (RuntimeTransfer *transfer : task->transfers) {

            while (!g_transfer_manager.completed(transfer)) {
                g_transfer_manager.progress(transfer);

                if (!g_transfer_manager.completed(transfer))
                    std::this_thread::yield();
            }

            task->pending_transfers.fetch_sub(
                1,
                std::memory_order_relaxed);
        }

        task->transfers_ready.store(
            true,
            std::memory_order_release);

        scheduler_pending.store(
            true,
            std::memory_order_release);

        /*
         * A transfer completion may make a compute task runnable while
         * no scheduler invocation is active. The scheduler itself already
         * serializes concurrent invocations through scheduler_running.
         */
        __kmp_dmdas_run_ready_tasks();
    }
}


void RuntimeExecutor::execute(RuntimeWorker *worker,
                              DMDASTask *task)
{

    /*
     * Compute execution must only receive tasks whose data transfers
     * have already completed in RuntimeTransferExecutor.
     */
    KMP_ASSERT(
        task->transfers_ready.load(
            std::memory_order_acquire));

    kmp_taskdata_t *taskdata =
        KMP_TASK_TO_TASKDATA(task->omp_task);

    // const int device = worker->device_id;



    /*__kmpc_setTaskStateWithDevice(
        taskdata->td_task_id,
        device,
        TaskState::SCHEDULED);*/

    task->submit_time =
        std::chrono::high_resolution_clock::now();

    task->submitted_to_openmp = true;

    kmp_int32 ret = __kmp_omp_task(
        task->gtid,
        task->omp_task,
        false);

    if (taskdata->td_flags.hidden_helper) {


        __kmp_hidden_helper_worker_thread_signal();
    }



}

RuntimeTransfer *
RuntimeTransferManager::find(
        RuntimeData *data,
        RuntimeNode *src,
        RuntimeNode *dst)
{


    TransferKey key{data, src, dst};

    auto it = active_transfers.find(key);

    if (it == active_transfers.end())
        return nullptr;

    RuntimeTransfer *transfer = it->second;

    if (transfer->state.load(
            std::memory_order_acquire) ==
        RuntimeTransfer::TRANSFER_FINISHED) {
        return nullptr;
    }

    return transfer;
}

void RuntimeTransferManager::clear_finished()
{
    std::lock_guard<std::mutex> lock(
        transfers_mutex);

    auto it = transfers.begin();

    while (it != transfers.end()) {

        RuntimeTransfer *transfer = *it;

        if (transfer->state.load(
                std::memory_order_acquire) ==
            RuntimeTransfer::TRANSFER_FINISHED) {

            TransferKey key{
                transfer->data,
                transfer->src,
                transfer->dst
            };

            auto active_it =
                active_transfers.find(key);

            if (active_it != active_transfers.end() &&
                active_it->second == transfer) {
                active_transfers.erase(active_it);
            }

            delete transfer;
            it = transfers.erase(it);

        } else {
            ++it;
        }
    }
}

RuntimeTransfer *
RuntimeTransferManager::submit(
    RuntimeData *data,
    RuntimeNode *src,
    RuntimeNode *dst)
{


    std::lock_guard<std::mutex> lock(
        transfers_mutex);

    RuntimeTransfer *existing =
        find(data, src, dst);

    if (existing != nullptr) {


        return existing;
    }



    RuntimeTransfer *t =
        new RuntimeTransfer{};

    t->data = data;
    t->src = src;
    t->dst = dst;

    t->completion_time =
        transfer_time(src, dst, data->size);

    t->finished.store(
        false,
        std::memory_order_relaxed);

    t->dma_worker = nullptr;
    t->submitted = true;
    t->started = false;

    t->state.store(
        RuntimeTransfer::TRANSFER_PENDING,
        std::memory_order_relaxed);

    transfers.push_back(t);

    TransferKey key{data, src, dst};

active_transfers[key] = t;

    return t;
}

static void execute_ompc_transfer(
    RuntimeData *data,
    RuntimeNode *src,
    RuntimeNode *dst)
{
    if (data == nullptr || src == nullptr || dst == nullptr)
        return;

    if (src == dst)
        return;

    using target_memtask_fn_t =
        void (*)(void *, int64_t, int64_t, int64_t);

    target_memtask_fn_t target_memtask = nullptr;

    *(void **)(&target_memtask) =
        KMP_DLSYM("__tgt_target_memtask");

    KMP_ASSERT(target_memtask != nullptr);




    target_memtask(
        data->addr,
        static_cast<int64_t>(data->size),
        static_cast<int64_t>(dst->id),
        static_cast<int64_t>(src->id));

}

void RuntimeTransferManager::progress(
    RuntimeTransfer *transfer)
{
    if (transfer == nullptr)
        return;

    RuntimeTransfer::RuntimeTransferState expected =
        RuntimeTransfer::TRANSFER_PENDING;

    /*
     * Only one task can transition the transfer from PENDING to RUNNING.
     * Other tasks observe that the transfer is already in progress.
     */
    if (!transfer->state.compare_exchange_strong(
            expected,
            RuntimeTransfer::TRANSFER_RUNNING)) {
        return;
    }

    /*
     * Do not hold dmdas_mutex while performing the MPI transfer.
     */
    execute_ompc_transfer(
        transfer->data,
        transfer->src,
        transfer->dst);

    /*
     * RuntimeData::copies is also accessed by callbacks and the scheduler,
     * so modifications must be protected by dmdas_mutex.
     */
    {
        std::lock_guard<std::mutex> lock(dmdas_mutex);

        bool found = false;

        for (RuntimeCopy &copy :
             transfer->data->copies) {
            if (copy.node == transfer->dst) {
                copy.state = RuntimeCopy::VALID;
                found = true;
                break;
            }
        }

        if (!found) {
            RuntimeCopy copy;
            copy.node = transfer->dst;
            copy.state = RuntimeCopy::VALID;

            transfer->data->copies.push_back(copy);
        }
    }

    transfer->finished.store(
        true,
        std::memory_order_release);

    transfer->state.store(
        RuntimeTransfer::TRANSFER_FINISHED,
        std::memory_order_release);



}

bool RuntimeTransferManager::completed(
    RuntimeTransfer *transfer)
{
    if (transfer == nullptr)
        return true;

    return transfer->state.load(
               std::memory_order_acquire) ==
           RuntimeTransfer::TRANSFER_FINISHED;
}

RuntimeData &
RuntimeDataManager::get(
    void *addr,
    size_t size)
{
    auto &data = data_map[addr];

    data.addr = addr;

    if (size > data.size)
        data.size = size;

    return data;
}
bool RuntimeDataManager::has_valid_copy(
    RuntimeData &data,
    RuntimeNode *node)
{
    for (auto &copy : data.copies)
    {
        if (copy.node == node && copy.state == RuntimeCopy::VALID)
            return true;
    }

    return false;
}


void RuntimeDataManager::write(
        RuntimeData &data,
        RuntimeNode *node)
{
    RuntimeCopy *existing = nullptr;

    for (auto &copy : data.copies)
    {
        if (copy.node == node)
            existing = &copy;

        copy.state = RuntimeCopy::INVALID;
    }

    if (existing)
    {
        existing->state = RuntimeCopy::VALID;
    }
    else
    {
        RuntimeCopy copy;

        copy.node = node;
        copy.state = RuntimeCopy::VALID;

        data.copies.push_back(copy);
    }

}

void RuntimeDataManager::read(
        RuntimeData &data,
        RuntimeNode *node)
{
    if (!has_valid_copy(data,node))
    {
        RuntimeNode *src = nullptr;

        for (auto &copy : data.copies)
        {
            if (copy.state == RuntimeCopy::VALID)
            {
                src = copy.node;
                break;
            }
        }

        if (src)
        {
            g_transfer_manager.submit(
                &data,
                src,
                node);
        }
    }
}

RuntimeNode *RuntimeDataManager::source(RuntimeData &data)
{

    for (auto &copy : data.copies)
    {
        if (copy.state == RuntimeCopy::VALID) {


            return copy.node;
        }
    }

    return nullptr;
}

void RuntimeScheduler::insert_sorted(
        RuntimeWorker *worker,
        DMDASTask *task)
{
    auto it = worker->ready_queue.begin();

    while (it != worker->ready_queue.end() &&
           (*it)->priority <= task->priority)
    {
        ++it;
    }

    worker->ready_queue.insert(it, task);
}

double RuntimeScheduler::compute_bottom_level(
        DMDASTask *task,
        std::unordered_map<DMDASTask*, bool> &visited)
{
    if (visited[task])
        return task->bottom_level;

    visited[task] = true;

    double best = 0.0;

    for (DMDASTask *succ : task->successors)
    {
        best = std::max(
                    best,
                    compute_bottom_level(
                        succ,
                        visited));
    }

    double average_compute_cost = 0.0;

    for (RuntimeWorker &worker : g_workers) {
        average_compute_cost +=
            g_perf_model.compute_cost(
                task,
                &worker);
    }

    if (!g_workers.empty()) {
        average_compute_cost /=
            static_cast<double>(g_workers.size());
    }

    task->bottom_level =
        average_compute_cost + best;

    return task->bottom_level;
}

void RuntimeScheduler::update_priorities()
{
    std::unordered_map<DMDASTask*, bool> visited;

    for (DMDASTask *task : g_dmdas_tasks)
        visited[task] = false;

    for (DMDASTask *task : g_dmdas_tasks)
        compute_bottom_level(task, visited);
}

void RuntimeScheduler::recompute_priorities()
{
    update_priorities();
}


void RuntimeScheduler::submit(DMDASTask *task)
{
    // update_priorities();

    RuntimeWorker *best_worker = nullptr;

    double best_score = DBL_MAX;
    double best_cost = 0.0;

    for (auto &worker : g_workers)
    {


        double compute_cost =
            g_perf_model.compute_cost(
                task,
                &worker);

        double transfer_cost = 0.0;

        double cost =
            worker.available_time +
            compute_cost;

        double locality_bonus = 0.0;

        for (size_t i = 0; i < task->deps.size(); ++i)
        {
            auto &dep = task->deps[i];

            if (!dep.flags.in)
                continue;

            void *addr =
                reinterpret_cast<void *>(dep.base_addr);

            size_t size = task->dep_sizes[i];

            RuntimeData &data =
                g_data_manager.get(addr, size);

            /*double transfer =
                g_data_manager.transfer_cost(
                    data,
                    worker.node);
             */

            RuntimeNode *src =
                g_data_manager.source(data);

            /*
             * If no proxy owns a copy, the data is still on the host.
             * Let libomptarget perform the initial host-to-device mapping.
             */
            if (src != nullptr) {

                double transfer =
                    g_data_manager.transfer_cost(
                        data,
                        worker.node);

                cost += transfer;
                transfer_cost += transfer;

                if (g_data_manager.has_valid_copy(
                        data,
                        worker.node)) {
                    locality_bonus += data.size;
                }
            }
        }

        double score =
            cost
            - task->bottom_level
            - locality_bonus * 1e-6;

        if (getenv("OMPCLUSTER_DMDAS_PLACEMENT_DEBUG")) {
            fprintf(
                stderr,
                "[DMDAS-PLACEMENT] task=%d model=%zx features=%zu "
                "worker=%d device=%d type=%d "
                "available=%.9g compute=%.9g transfer=%.9g "
                "locality_bytes=%.0f bottom=%.9g score=%.9g\n",
                task->instance_id,
                task->model_id,
                task->perf_features.values.size(),
                worker.id,
                worker.device_id,
                worker.device_type,
                worker.available_time,
                compute_cost,
                transfer_cost,
                locality_bonus,
                task->bottom_level,
                score);
        }

        if (score < best_score)
        {
            best_score = score;
            best_cost = cost;
            best_worker = &worker;
        }
    }

    /*
     * Calibration-only worker override.
     *
     * This is intentionally applied after normal DMDAS scoring so the
     * production scheduling policy remains unchanged unless the explicit
     * environment variable is set.
     *
     * GPU1 mapping:
     *   worker 0 = CPU
     *   worker 1 = CUDA / P100
     */
    if (const char *force_env =
            getenv("OMPCLUSTER_DMDAS_FORCE_WORKER")) {

        int forced_worker = atoi(force_env);

        if (forced_worker >= 0 &&
            static_cast<size_t>(forced_worker) < g_workers.size()) {

            best_worker = &g_workers[forced_worker];

            /*
             * Recompute the selected worker's predicted availability.
             * This value is only bookkeeping during calibration; the
             * measured execution time remains the value recorded by the
             * performance model callback.
             */
            best_cost = best_worker->available_time;
            best_score = best_cost - task->bottom_level;

            if (getenv("OMPCLUSTER_DMDAS_PLACEMENT_DEBUG")) {
                fprintf(
                    stderr,
                    "[DMDAS-FORCE-WORKER] task=%d model=%zx "
                    "worker=%d device=%d type=%d\\n",
                    task->instance_id,
                    task->model_id,
                    best_worker->id,
                    best_worker->device_id,
                    best_worker->device_type);
            }
        }
    }

    if (getenv("OMPCLUSTER_DMDAS_PLACEMENT_DEBUG") &&
        best_worker != nullptr) {
        fprintf(
            stderr,
            "[DMDAS-SELECT] task=%d model=%zx features=%zu "
            "worker=%d device=%d type=%d "
            "cost=%.9g score=%.9g\n",
            task->instance_id,
            task->model_id,
            task->perf_features.values.size(),
            best_worker->id,
            best_worker->device_id,
            best_worker->device_type,
            best_cost,
            best_score);
    }

    if (task->model_id != 0 && task->transfers.empty()) {
        for (size_t i = 0; i < task->deps.size(); ++i) {
            auto &dep = task->deps[i];
            if (!dep.flags.in)
                continue;

            void *addr =
                reinterpret_cast<void *>(dep.base_addr);

            size_t size = task->dep_sizes[i];

            RuntimeData &data =
                g_data_manager.get(addr, size);

            /*
            * No transfer is required if the destination node
            * already owns a valid copy.
            */
            if (g_data_manager.has_valid_copy(
                    data,
                    best_worker->node)) {
                continue;
            }

            RuntimeNode *src =
                g_data_manager.source(data);

            if (src != nullptr) {

                RuntimeTransfer *transfer =
                    g_transfer_manager.submit(
                        &data,
                        src,
                        best_worker->node);

                task->transfers.push_back(transfer);


            }
        }
    }

    /*
     * Async-transfer bookkeeping.
     *
     * For now this does not alter scheduling behavior. A task with no
     * required transfer is immediately data-ready. Otherwise its transfer
     * engine state starts as not-yet-started.
     */
    const bool has_transfers =
        !task->transfers.empty();

    task->pending_transfers.store(
        static_cast<int>(task->transfers.size()),
        std::memory_order_relaxed);

    task->transfers_started.store(
        has_transfers,
        std::memory_order_relaxed);

    task->transfers_ready.store(
        !has_transfers,
        std::memory_order_release);

    /*
     * Start data movement independently from compute scheduling.
     * The compute worker is NOT marked busy here.
     */
    if (has_transfers) {
        g_transfer_executor.submit(task);
    }

    if (best_worker == nullptr) {
        printf("DMDAS ERROR: no worker selected for task %d\n", task->instance_id);
        return;
    }

    best_worker->available_time = best_cost;

    task->priority = best_score;

    if (task->model_id != 0 && best_worker != nullptr) {
    if (best_worker->id >= 0 &&
        static_cast<size_t>(best_worker->id) < g_compute_placement.size()) {

        g_compute_placement[best_worker->id]++;
    }
}

    // insert_sorted(best_worker, task);
    best_worker->ready_queue.push_back(task);

    task->submitted = true;

}



DMDASTask *
RuntimeScheduler::pick_next(
        RuntimeWorker *worker)
{
    if (worker->busy)
        return nullptr;

    if (worker->ready_queue.empty()) {
        return nullptr;
    }

    /*
     * A task may be dependency-ready but still waiting for its data.
     *
     * Do not reserve the compute worker for such a task. Also avoid
     * head-of-line blocking: a later task whose transfers have completed
     * may run first.
     */
    for (auto it = worker->ready_queue.begin();
         it != worker->ready_queue.end();
         ++it) {

        DMDASTask *task = *it;

        if (!task->transfers_ready.load(
                std::memory_order_acquire)) {
            continue;
        }

        worker->ready_queue.erase(it);

        return task;
    }

    /*
     * The queue contains work, but none of it is data-ready yet.
     */
    return nullptr;
}

static void build_default_perf_features(
    DMDASTask *task)
{
    if (!task)
        return;

    // Explicit application features have priority.
    if (!task->perf_features.empty())
        return;

    double data_bytes = 0.0;

    for (size_t i = 0;
         i < task->dep_sizes.size();
         ++i) {
        data_bytes +=
            static_cast<double>(
                task->dep_sizes[i]);
    }

    task->perf_features.add(
        PerfFeatureKind::DATA_BYTES,
        data_bytes);
}




double PerformanceModel::compute_cost(
        DMDASTask *task,
        RuntimeWorker *worker)
{


    /*
     * model_id == 0 means that this task has no identifiable target
     * kernel (no td_target_info / outlined_fn_id).
     *
     * Do not inject an arbitrary 10-second compute cost into the
     * worker availability timeline.  Keep an environment override
     * for controlled ablation/reproduction of the legacy behaviour.
     */
    if (task->model_id == 0) {
        const char *env =
            getenv("OMPCLUSTER_DMDAS_NON_TARGET_COST");

        if (env != nullptr)
            return atof(env);

        return 0.0;
    }

    build_default_perf_features(task);

    if (task->perf_features.empty())
        return 10.0; // ton fallback actuel

    std::vector<double> features =
        get_feature_values(
            task->perf_features);

    size_t feature_signature =
        get_feature_signature(
            task->perf_features);

    PerfModelKey key {
        task->model_id,
        worker->id,
        feature_signature,
        task->perf_features.values.size()
    };


    auto model_it =
        models.find(key);

    double prediction =
        (model_it == models.end())
            ? 0.004
            : model_it->second.predict(features);

    return prediction;

}

double PerformanceModel::transfer_cost(
    RuntimeData *data,
    RuntimeWorker *worker)
{
    return g_data_manager.transfer_cost(
                *data,
                worker->node);
}

void PerformanceModel::record_execution(
        DMDASTask *task,
        RuntimeWorker *worker,
        double duration)
{

    build_default_perf_features(task);

    if (task->perf_features.empty())
        return;

    std::vector<double> features =
        get_feature_values(
            task->perf_features);

    size_t feature_signature =
        get_feature_signature(
            task->perf_features);

    PerfModelKey key {
        task->model_id,
        worker->id,
        feature_signature,
        task->perf_features.values.size()
    };


        /*
     * Test:
     * learn only during the first execution.
     */
    if (!g_perf_model_frozen.load(
            std::memory_order_relaxed)) {

        models[key].update(
            features,
            duration);
    }



    std::string sample_path =
        get_perf_sample_path();

    if (!sample_path.empty()) {
        save_sample(
            sample_path.c_str(),
            task,
            worker,
            duration);
    }
}

static double transfer_time(
    RuntimeNode *src,
    RuntimeNode *dst,
    size_t bytes)
{
    if (src == dst)
        return 0.0;

    if (dst->id < 0 ||
        static_cast<size_t>(dst->id) >= src->bandwidth.size())
        return 0.0;

    double bandwidth =
        src->bandwidth[dst->id];

    double latency =
        src->latency[dst->id];

    if (bandwidth <= 0.0)
        return 0.0;

    return latency +
           static_cast<double>(bytes) /
           bandwidth;
}

double RuntimeDataManager::transfer_cost(
        RuntimeData &data,
        RuntimeNode *node)
{
    double best = DBL_MAX;

    for (auto &copy : data.copies)
    {
        if (copy.state != RuntimeCopy::VALID)
            continue;

        double cost =
            transfer_time(
                copy.node,
                node,
                data.size);

        best = std::min(best, cost);
    }

    return best;
}


void RuntimeScheduler::task_completed(DMDASTask *task)
{
    if (task->completed)
        return;

    task->completed = true;

    for (DMDASTask *succ : task->successors)
    {
        int remaining =
            succ->remaining_dependencies.fetch_sub(1) - 1;

        if (remaining == 0 && succ->native_ready)
            submit(succ);
    }
}

void __kmp_dmdas_native_ready(kmp_task_t *omp_task)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_map.find(omp_task);

    if (it == g_task_map.end())
        return;

    DMDASTask *task = it->second;

    task->native_ready = true;

    if (task->remaining_dependencies.load() == 0 &&
        !task->submitted) {
        g_scheduler.submit(task);
        scheduler_pending.store(true);
    }
}

void __kmp_dmdas_init(int /*nworkers*/)
{
    int ndevices = omp_get_num_devices();

    if (ndevices <= 0) {
        printf("DMDAS ERROR: no OMPC devices detected\n");
        return;
    }

    g_workers.clear();
    g_nodes.clear();

    g_workers.resize(ndevices);
    g_nodes.resize(ndevices);

    double default_bandwidth = 1.0e9;
    double default_latency = 0.0;

    if (const char *env =
            getenv("OMPCLUSTER_DMDAS_BANDWIDTH")) {
        default_bandwidth = atof(env);
    }

    if (const char *env =
            getenv("OMPCLUSTER_DMDAS_LATENCY")) {
        default_latency = atof(env);
    }

    for (int device = 0;
         device < ndevices;
         ++device) {

        RuntimeNode &node =
            g_nodes[device];

        RuntimeWorker &worker =
            g_workers[device];

        node.id = device;
        node.bandwidth.assign(
            ndevices, default_bandwidth);
        node.latency.assign(
            ndevices, default_latency);

        /*
         * Communication parameters can be overridden for each
         * directional device pair:
         *
         * OMPCLUSTER_DMDAS_BANDWIDTH_<SRC>_<DST>
         * OMPCLUSTER_DMDAS_LATENCY_<SRC>_<DST>
         */
        for (int dst = 0; dst < ndevices; ++dst) {
            if (dst == device) {
                node.bandwidth[dst] = default_bandwidth;
                node.latency[dst] = 0.0;
                continue;
            }

            char name[128];

            snprintf(
                name, sizeof(name),
                "OMPCLUSTER_DMDAS_BANDWIDTH_%d_%d",
                device, dst);

            if (const char *env = getenv(name))
                node.bandwidth[dst] = atof(env);

            snprintf(
                name, sizeof(name),
                "OMPCLUSTER_DMDAS_LATENCY_%d_%d",
                device, dst);

            if (const char *env = getenv(name))
                node.latency[dst] = atof(env);
        }

        node.workers.clear();

        worker.id = device;
        worker.device_id = device;
        worker.device_type =
            __kmpc_get_device_type(device);
        worker.node = &node;

        worker.busy = false;
        worker.current_task = nullptr;
        worker.available_time = 0.0;
        worker.ready_queue.clear();

        node.workers.push_back(&worker);
    }

    g_compute_placement.assign(g_workers.size(), 0);

    /*
     * Start the persistent transfer workers used for asynchronous
     * data transfers.
     */
    g_transfer_executor.start(8);

    std::string perf_model_path =
        get_perf_model_path();

    if (!perf_model_path.empty()) {
        int loaded =
            g_perf_model.load(
                perf_model_path.c_str());

        if (loaded > 0) {
            /*
             * By default, preserve the existing behaviour: a model loaded
             * from disk is frozen and used only for prediction.
             *
             * In explicit training mode, however, keep the loaded
             * sufficient statistics (n, X^T X, X^T y, ...) trainable so
             * that subsequent executions can extend the model.
             */
            const char *training_env =
                getenv("OMPCLUSTER_PERF_MODEL_TRAINING");

            bool training =
                training_env != nullptr &&
                strcmp(training_env, "0") != 0;

            g_perf_model_frozen.store(
                !training,
                std::memory_order_relaxed);

            printf(
                "PERF MODEL: loaded %d entries (%s)\n",
                loaded,
                training
                    ? "training continues"
                    : "frozen");
        }
    }
}



bool __kmp_use_dmdas_runtime()
{
    const char *env = getenv("OMPCLUSTER_RUNTIME");



    return env && strcmp(env, "dmdas") == 0;
}

static void save_and_freeze_perf_model_locked()
{
    bool was_frozen =
        g_perf_model_frozen.exchange(
            true,
            std::memory_order_relaxed);

    if (was_frozen)
        return;

    std::string perf_model_path =
        get_perf_model_path();

    if (!perf_model_path.empty()) {
        g_perf_model.save(
            perf_model_path.c_str());
    }
}

void __kmpc_dmdas_save_perf_model()
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);
    save_and_freeze_perf_model_locked();
}

void __kmp_dmdas_reset_for_replay()
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    static int replay = 0;

    ++replay;

    if (replay == 1) {
        const char *training_env =
            getenv("OMPCLUSTER_PERF_MODEL_TRAINING");

        bool training =
            training_env != nullptr &&
            strcmp(training_env, "0") != 0;

        if (training) {
            /*
             * Training mode spans TDG replays.  Save the current
             * sufficient statistics, but keep the model trainable so
             * later executions continue updating the same model.
             */
            std::string perf_model_path =
                get_perf_model_path();

            if (!perf_model_path.empty()) {
                g_perf_model.save(
                    perf_model_path.c_str());
            }
        } else {
            /*
             * Default behaviour: learn from the first execution,
             * then freeze the model for subsequent TDG replays.
             */
            save_and_freeze_perf_model_locked();
        }
    }

    /*
     * OpenMP tasks are reused during TDG replay, so only the DMDAS
     * runtime state needs to be reset.
     */
    for (RuntimeWorker &worker : g_workers) {
        worker.ready_queue.clear();
        worker.current_task = nullptr;
        worker.busy = false;
        worker.available_time = 0.0;
    }

    for (DMDASTask *task : g_dmdas_tasks) {
        task->completed = false;
        task->submitted = false;
        task->assigned_worker = nullptr;
        task->transfers.clear();

        /*
         * Re-arm asynchronous transfer state for TDG replay.
         * RuntimeScheduler::submit() will rebuild the transfer list
         * and set the definitive pending/ready state.
         */
        task->pending_transfers.store(
            0,
            std::memory_order_relaxed);

        task->transfers_started.store(
            false,
            std::memory_order_relaxed);

        task->transfers_ready.store(
            false,
            std::memory_order_relaxed);

        task->remaining_dependencies.store(
            static_cast<int>(
                task->predecessors.size()));
    }

    /*
     * Tasks no longer reference RuntimeTransfer objects from previous
     * executions.
     */
    g_transfer_manager.clear_finished();
    std::fill(g_compute_placement.begin(),
          g_compute_placement.end(),
          0);

    /*
 * TEST:
 * replay starts without DMDAS locality history
 * from the previous execution.
 */
    // g_data_manager.reset_copies(); // TEMP ABLATION


    /*
     * Recompute priorities once per replay,
     * not once per submitted task.
     */
    g_scheduler.recompute_priorities();

    /*
     * Submit root tasks.
     */
    for (DMDASTask *task : g_dmdas_tasks) {
        if (task->remaining_dependencies.load() == 0) {
            g_scheduler.submit(task);
        }
    }
}

void __kmp_dmdas_run_ready_tasks()
{
    bool expected = false;

    if (!scheduler_running.compare_exchange_strong(
            expected, true)) {
        scheduler_pending.store(true);
        return;
    }

    while (true) {
        scheduler_pending.store(false);

        std::vector<
            std::pair<RuntimeWorker *, DMDASTask *>
        > to_execute;

        /*
         * Only select compute tasks whose transfers have already
         * completed. RuntimeScheduler::pick_next() enforces the
         * transfers_ready condition.
         *
         * No data movement is performed while holding dmdas_mutex.
         */
        {
            std::lock_guard<std::mutex> lock(dmdas_mutex);

            for (RuntimeWorker &worker : g_workers) {
                DMDASTask *task =
                    g_scheduler.pick_next(&worker);

                if (task == nullptr)
                    continue;

                /*
                 * worker.busy now means compute reserved/running.
                 * Transfer execution happens independently in
                 * RuntimeTransferExecutor.
                 */
                worker.busy = true;
                worker.current_task = task;
                task->assigned_worker = &worker;

                to_execute.emplace_back(
                    &worker,
                    task);
            }
        }

        /*
         * Submit compute outside dmdas_mutex.
         *
         * Transfers are already complete for every task selected above,
         * so RuntimeExecutor::execute() should not block on data movement.
         * Its old transfer wait is temporarily retained as a safety check.
         */
        for (auto &[worker, task] : to_execute) {
            g_executor.execute(worker, task);
        }

        /*
         * A task completion or transfer completion arriving during this
         * pass requests another inspection of the worker queues.
         */
        if (!scheduler_pending.exchange(false))
            break;
    }

    {
        std::lock_guard<std::mutex> lock(dmdas_mutex);
    }

    scheduler_running.store(false);

    /*
     * Close the race where a completion sets scheduler_pending just
     * before scheduler_running becomes false.
     */
    if (scheduler_pending.load())
        __kmp_dmdas_run_ready_tasks();
}

static void __kmp_dmdas_task_completed(DMDASTask *task)
{
    g_scheduler.task_completed(task);
}

static size_t get_target_arg_size(
    DMDASTask *task,
    const kmp_depend_info_t &dep)
{
    kmp_taskdata_t *taskdata =
        KMP_TASK_TO_TASKDATA(task->omp_task);

    kmp_target_task_data_t *target =
        taskdata->td_target_info;

    void *dep_addr =
        reinterpret_cast<void *>(dep.base_addr);

    if (target == nullptr)
        return dep.len;

    for (int i = 0; i < target->num_args; ++i) {
        if (target->args[i] == dep_addr) {
            size_t arg_size =
                static_cast<size_t>(target->arg_sizes[i]);

            if (arg_size > 0)
                return arg_size;

            return dep.len;
        }
    }

    return dep.len;
}

void __kmpc_dmdas_record_kernel_time(
    int32_t task_id,
    double kernel_time)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_id_map.find(task_id);

    if (it == g_task_id_map.end()) {

        return;
    }

    DMDASTask *task = it->second;

    if (task->assigned_worker == nullptr) {

        return;
    }

    g_perf_model.record_execution(
        task,
        task->assigned_worker,
        kernel_time);
}

void __kmp_dmdas_notify_task_started(kmp_task_t *omp_task)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_map.find(omp_task);

    if (it == g_task_map.end())
        return;

    DMDASTask *task = it->second;

    if (task->model_id == 0)
        return;

    task->execution_start =
        std::chrono::high_resolution_clock::now();

    task->execution_started = true;
}

void __kmp_dmdas_notify_target_completed(int32_t task_id)
{
return;
/*

    std::lock_guard<std::mutex> lock(dmdas_mutex);

    auto it = g_task_id_map.find(task_id);

    if (it == g_task_id_map.end()) {

        return;
    }

    DMDASTask *task = it->second;


    if (!task->target_started ||
        task->assigned_worker == nullptr)
        return;

    auto target_end =
        std::chrono::high_resolution_clock::now();

    double target_duration =
        std::chrono::duration<double>(
            target_end -
            task->target_start
        ).count();

    task->target_started = false;
    */
}

void __kmp_dmdas_notify_task_completed(kmp_task_t *omp_task)
{
    bool launch_scheduler = false;
    {

        std::lock_guard<std::mutex> lock(dmdas_mutex);

        auto it = g_task_map.find(omp_task);

        if (it == g_task_map.end()) {
            return;
        }

        DMDASTask *task = it->second;

        /*
        * Record the execution time before modifying the worker state.
        *
        * The timestamp was recorded when DMDAS submitted the task.
        * At this point OpenMP/libomptarget has reported its completion.
        */
        if (task->assigned_worker != nullptr) {
            RuntimeWorker *worker =
                task->assigned_worker;

            RuntimeNode *node =
                worker->node;

            /*
             * Target execution time is recorded through
             * __kmpc_dmdas_record_kernel_time() for both CPU and GPU
             * target devices.  Do not record it again here.
             */

            for (size_t i = 0; i < task->deps.size(); ++i) {

                    auto &dep = task->deps[i];

                    if (!dep.flags.out)
                        continue;


                    void *addr =
                        reinterpret_cast<void *>(dep.base_addr);

                    size_t size = task->dep_sizes[i];

                    RuntimeData &data =
                        g_data_manager.get(addr, size);

                    g_data_manager.write(data, node);

                }


            task->assigned_worker->current_task = nullptr;
            task->assigned_worker->busy = false;
        }

        /*
         * Output locations are updated before successors become ready.
         */
        g_scheduler.task_completed(task);

        /*
         * A task has completed, so a worker is available and new tasks
         * may have become ready.
         */
        scheduler_pending.store(true);
        launch_scheduler = !scheduler_running.load();

    }

    if (launch_scheduler)
        __kmp_dmdas_run_ready_tasks();

}

static void add_edge(DMDASTask *pred,
                     DMDASTask *task)
{
    if (!pred || pred == task)
        return;

    // Éviter la même arête deux fois.
    if (std::find(task->predecessors.begin(),
                  task->predecessors.end(),
                  pred) != task->predecessors.end()) {
        return;
    }

    pred->successors.push_back(task);
    task->predecessors.push_back(pred);

    if (!pred->completed)
        task->remaining_dependencies.fetch_add(1);

}

kmp_int32 __kmp_dmdas_task_with_deps(
    ident_t *loc_ref,
    kmp_int32 gtid,
    kmp_task_t *new_task,
    kmp_int32 ndeps,
    kmp_depend_info_t *dep_list,
    kmp_int32 ndeps_noalias,
    kmp_depend_info_t *noalias_dep_list)
{
    std::lock_guard<std::mutex> lock(dmdas_mutex);

    static bool initialized = false;

    if (!initialized) {
        kmp_info_t *thread = __kmp_threads[gtid];
        kmp_team_t *team = thread->th.th_team;
        int workers = team->t.t_nproc;


        __kmp_dmdas_init(workers);
        initialized = true;
    }


    // Prepare the DMDAS task context.
    DMDASTask *task = new DMDASTask;
    task->remaining_dependencies.store(0);
    task->submitted = false;
    task->completed = false;

    task->gtid = gtid;

    auto pending_it =
        g_pending_perf_features.find(gtid);

    if (pending_it !=
        g_pending_perf_features.end()) {

        task->perf_features =
            std::move(pending_it->second);

        g_pending_perf_features.erase(
            pending_it);
    }
    task->omp_task = new_task;
    task->current_task = __kmp_threads[gtid]->th.th_current_task;
    task->instance_id = g_dmdas_tasks.size() + 1;



kmp_taskdata_t *taskdata =
    KMP_TASK_TO_TASKDATA(new_task);

    g_task_id_map[taskdata->td_task_id] = task;

if (taskdata->td_target_info != nullptr &&
    taskdata->td_target_info->outlined_fn_id != nullptr) {

    /*task->model_id =
        reinterpret_cast<uintptr_t>(
            taskdata->td_target_info->outlined_fn_id);*/

    task->model_id =
        get_stable_model_id(
            taskdata->td_target_info->outlined_fn_id);

} else {
    task->model_id = 0;

}


    for (int i = 0; i < ndeps; ++i) {
        task->deps.push_back(dep_list[i]);

        size_t size =
            get_target_arg_size(
                task,
                task->deps.back());

        task->dep_sizes.push_back(size);

    }

    g_dmdas_tasks.push_back(task);

    for (int i = 0; i < ndeps; i++)
    {
        kmp_depend_info_t &dep = task->deps[i];

        void *addr =
            reinterpret_cast<void *>(dep.base_addr);

        size_t size = task->dep_sizes[i];

        RuntimeData &data =
            g_data_manager.get(addr, size);

        /*
         * Another task may already have provided a more precise size for
         * the same data. RuntimeData::size stores the largest canonical
         * size known for this address.
         */
        task->dep_sizes[i] = data.size;


    }




    /*
     * NOTE: The native OpenMP dependency tracker treats an OUT/INOUT
     * following a set of readers differently from this simplified DMDAS
     * dependency model. Native links the new OUT/INOUT task to the active
     * reader set only, whereas DMDAS currently links it to both the last
     * writer and active readers. This may introduce a redundant transitive
     * edge (observed as 1025 DMDAS predecessors vs 1024 native predecessors
     * in TaskBench 32x32). It did not affect the observed TaskBench
     * execution, but should be revisited if exact dependency-graph
     * equivalence is required.
     */
    for (int i = 0; i < ndeps; ++i) {
        kmp_depend_info_t &dep = task->deps[i];

        void *addr =
            reinterpret_cast<void *>(dep.base_addr);

        bool is_read =
            dep.flags.in && !dep.flags.out;

        bool is_write =
            dep.flags.out;

        /*
         * A read-only dependency must wait for the last writer. The task
         * is then recorded as an active reader.
         */
        if (is_read) {
            auto writer = g_last_writer.find(addr);

            if (writer != g_last_writer.end())
                add_edge(writer->second, task);

            g_last_readers[addr].push_back(task);
        }

        /*
         * Write or inout dependency.
         *
         * A write must wait for the last writer and all active readers.
         */
        if (is_write) {
            auto writer = g_last_writer.find(addr);

            if (writer != g_last_writer.end())
                add_edge(writer->second, task);

            auto readers = g_last_readers.find(addr);

            if (readers != g_last_readers.end()) {
                for (DMDASTask *reader : readers->second)
                    add_edge(reader, task);

                readers->second.clear();
            }

            g_last_writer[addr] = task;
        }
    }

    g_task_map[new_task] = task;

    if (task->remaining_dependencies.load() == 0 &&
        task->native_ready &&
        !task->submitted) {
        g_scheduler.submit(task);
        scheduler_pending.store(true);
    }



    return TASK_CURRENT_NOT_QUEUED;
}
