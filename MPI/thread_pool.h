#pragma once
#include <pthread.h>
#include <atomic>
#include <algorithm>
#include <functional>
#include <vector>
#include <cstddef>

namespace tp {

inline std::atomic<int>& parallel_depth() {
    static std::atomic<int> d{0};
    return d;
}

struct ParallelGuard {
    ParallelGuard() { parallel_depth().fetch_add(1, std::memory_order_relaxed); }
    ~ParallelGuard() { parallel_depth().fetch_sub(1, std::memory_order_relaxed); }
};

inline bool in_parallel() {
    return parallel_depth().load(std::memory_order_relaxed) > 0;
}

inline int& num_threads_storage() {
    static int n = 1;
    return n;
}

inline bool should_parallel(size_t work_units, int nt) {
    return nt > 1 && work_units >= static_cast<size_t>(nt) * 4;
}

inline bool should_parallel_tasks(size_t tasks, int nt) {
    return nt > 1 && tasks > 1;
}

struct WorkerPool {
    struct WorkerArg {
        WorkerPool* pool;
        int tid;
    };

    int nt = 0;
    std::atomic<bool> stop{false};
    std::function<void(int)>* job = nullptr;
    pthread_barrier_t barrier_start{};
    pthread_barrier_t barrier_end{};
    std::vector<pthread_t> workers;
    std::vector<WorkerArg> worker_args;

    static void* worker_entry(void* arg) {
        WorkerArg* wa = static_cast<WorkerArg*>(arg);
        WorkerPool* self = wa->pool;
        const int tid = wa->tid;
        for (;;) {
            pthread_barrier_wait(&self->barrier_start);
            if (self->stop.load(std::memory_order_relaxed)) {
                pthread_barrier_wait(&self->barrier_end);
                return nullptr;
            }
            if (self->job) (*self->job)(tid);
            pthread_barrier_wait(&self->barrier_end);
        }
    }

    void destroy() {
        if (nt <= 1) return;
        stop.store(true, std::memory_order_relaxed);
        pthread_barrier_wait(&barrier_start);
        pthread_barrier_wait(&barrier_end);
        for (pthread_t& th : workers) pthread_join(th, nullptr);
        workers.clear();
        worker_args.clear();
        pthread_barrier_destroy(&barrier_start);
        pthread_barrier_destroy(&barrier_end);
        nt = 0;
        stop.store(false, std::memory_order_relaxed);
    }

    void ensure(int n) {
        n = std::max(1, n);
        if (n == nt) return;
        destroy();
        nt = n;
        if (nt == 1) return;
        pthread_barrier_init(&barrier_start, nullptr, static_cast<unsigned>(nt));
        pthread_barrier_init(&barrier_end, nullptr, static_cast<unsigned>(nt));
        workers.resize(static_cast<size_t>(nt - 1));
        worker_args.resize(static_cast<size_t>(nt - 1));
        for (int t = 1; t < nt; ++t) {
            worker_args[static_cast<size_t>(t - 1)] = {this, t};
            pthread_create(&workers[static_cast<size_t>(t - 1)], nullptr, worker_entry,
                           &worker_args[static_cast<size_t>(t - 1)]);
        }
    }

    void run(std::function<void(int)>& fn) {
        if (nt <= 1) {
            fn(0);
            return;
        }
        job = &fn;
        pthread_barrier_wait(&barrier_start);
        fn(0);
        pthread_barrier_wait(&barrier_end);
        job = nullptr;
    }
};

inline WorkerPool& pool() {
    static WorkerPool p;
    return p;
}

inline void set_num_threads(int n) {
    num_threads_storage() = std::max(1, n);
    pool().ensure(num_threads_storage());
}

inline int get_num_threads() {
    return num_threads_storage();
}

inline void shutdown_pool() {
    pool().destroy();
}

template<typename Fn>
inline void parallel_region(Fn&& fn) {
    if (in_parallel()) {
        fn(0);
        return;
    }
    const int nt = get_num_threads();
    if (nt <= 1) {
        fn(0);
        return;
    }
    ParallelGuard guard;
    pool().ensure(nt);
    std::function<void(int)> job = std::ref(fn);
    pool().run(job);
}

template<typename Fn>
inline void parallel_for_static(size_t begin, size_t end, Fn&& fn) {
    if (end <= begin) return;
    const int nt = get_num_threads();
    if (in_parallel() || nt <= 1 || !should_parallel(end - begin, nt)) {
        for (size_t i = begin; i < end; ++i) fn(i);
        return;
    }
    parallel_region([&](int tid) {
        const size_t total = end - begin;
        const size_t chunk = (total + static_cast<size_t>(nt) - 1) / static_cast<size_t>(nt);
        const size_t b = begin + static_cast<size_t>(tid) * chunk;
        const size_t e = std::min(end, b + chunk);
        for (size_t i = b; i < e; ++i) fn(i);
    });
}

template<typename Fn>
inline void parallel_for_dynamic(size_t count, Fn&& fn) {
    if (count == 0) return;
    const int nt = get_num_threads();
    if (in_parallel() || nt <= 1 || !should_parallel(count, nt)) {
        for (size_t i = 0; i < count; ++i) fn(i);
        return;
    }
    ParallelGuard guard;
    std::atomic<size_t> cursor{0};
    parallel_region([&](int) {
        for (;;) {
            size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) break;
            fn(i);
        }
    });
}

template<typename Fn>
inline void parallel_for_dynamic_tid(size_t count, Fn&& fn) {
    if (count == 0) return;
    const int nt = get_num_threads();
    if (in_parallel() || nt <= 1 || !should_parallel_tasks(count, nt)) {
        for (size_t i = 0; i < count; ++i) fn(0, i);
        return;
    }
    ParallelGuard guard;
    std::atomic<size_t> cursor{0};
    parallel_region([&](int tid) {
        for (;;) {
            size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) break;
            fn(tid, i);
        }
    });
}

}  // namespace tp