#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <mpi.h> 

#include "kmeans.h"
#include "profiler.h"
#include "simd_l2.h"
#include "aligned_alloc.h"
#include "thread_pool.h"
#include "hnswlib/hnswlib.h"

class IVFHNSWIndex {
public:
    int d;
    int n_lists;
    int hnsw_m;
    int hnsw_ef_construction;

    AlignedVector<float> ivf_centroids; 
    
    hnswlib::L2Space* space = nullptr;
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_graphs;

    IVFHNSWIndex(int dim = 96, int nlist = 1024, int m = 16, int ef_construction = 200) 
        : d(dim), n_lists(nlist), hnsw_m(m), hnsw_ef_construction(ef_construction) {
        
        ivf_centroids.resize(n_lists * d);
        hnsw_graphs.resize(n_lists, nullptr);
        
        space = new hnswlib::L2Space(d);
    }

    ~IVFHNSWIndex() {
        for (auto* graph : hnsw_graphs) {
            if (graph != nullptr) {
                delete graph;
            }
        }
        if (space != nullptr) {
            delete space;
        }
    }

    void build(const float* local_data, size_t local_n, int rank, int size) {
        if (rank == 0) {
            std::cerr << "[Rank 0] [IVF-HNSW Build] d=" << d << ", n_lists=" << n_lists 
                      << ", HNSW_M=" << hnsw_m << ", HNSW_ef_c=" << hnsw_ef_construction << "\n";
        }

        // 1. 全局 IVF 粗聚类中心训练与广播 (逻辑保留自 IVF-PQ)
        {
            MicroProfiler::Timer _t("1_Train_IVF_&_Bcast");
            
            int sample_per_node = std::min(static_cast<size_t>(25000), local_n);
            std::vector<float> local_sample(sample_per_node * d);
            
            for(int i = 0; i < sample_per_node; ++i) {
                size_t idx = (local_n / sample_per_node) * i; 
                std::memcpy(&local_sample[i * d], &local_data[idx * d], d * sizeof(float));
            }

            std::vector<float> global_sample;
            if (rank == 0) {
                global_sample.resize(sample_per_node * size * d);
            }

            MPI_Gather(local_sample.data(), sample_per_node * d, MPI_FLOAT,
                       global_sample.data(), sample_per_node * d, MPI_FLOAT,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                KMeans ivf_km(d, n_lists);
                ivf_km.train(global_sample.data(), sample_per_node * size);
                std::memcpy(ivf_centroids.data(), ivf_km.centroids.data(), n_lists * d * sizeof(float));
            }
            MPI_Bcast(ivf_centroids.data(), n_lists * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // 2. 局部数据 IVF 桶分配及计数
        std::vector<int> assign(local_n);
        std::vector<size_t> bucket_counts(n_lists, 0);

        {
            MicroProfiler::Timer _t("2_Assign_IVF_Buckets");
            int num_threads = tp::get_num_threads();
            std::vector<std::vector<size_t>> thread_local_counts(
                static_cast<size_t>(num_threads), std::vector<size_t>(n_lists, 0));

            tp::parallel_region([&](int tid) {
                const size_t chunk = (local_n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
                const size_t i0 = static_cast<size_t>(tid) * chunk;
                const size_t i1 = std::min(local_n, i0 + chunk);

                for (size_t i = i0; i < i1; ++i) {
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;
                    for (int c = 0; c < n_lists; ++c) {
                        float dist = compute_L2_sqr(local_data + i * d, &ivf_centroids[c * d], d);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_c = c;
                        }
                    }
                    assign[i] = best_c;
                    thread_local_counts[static_cast<size_t>(tid)][best_c]++;
                }
            });

            // 归并各个线程的统计数据
            for (int t = 0; t < num_threads; ++t) {
                for (int c = 0; c < n_lists; ++c) {
                    bucket_counts[c] += thread_local_counts[static_cast<size_t>(t)][c];
                }
            }
        }

        // 3. 根据分配数量并行初始化 HNSW 图结构
        {
            MicroProfiler::Timer _t("3_Init_HNSW_Graphs");
            for (int c = 0; c < n_lists; ++c) {
                if (bucket_counts[c] > 0) {
                    // 初始化对应的局部小世界图，预分配该桶对应的内存容量
                    hnsw_graphs[c] = new hnswlib::HierarchicalNSW<float>(
                        space, bucket_counts[c], hnsw_m, hnsw_ef_construction);
                }
            }
        }

        // 4. 利用多线程并行插入底库向量到对应的局部 HNSW 图中
        {
            MicroProfiler::Timer _t("4_Build_HNSW_Graphs");
            tp::parallel_for_static(0, local_n, [&](size_t i) {
                int c = assign[i];
                hnsw_graphs[c]->addPoint((const void*)(local_data + i * d), static_cast<hnswlib::labeltype>(i));
            });
        }
        
        if (rank == 0) {
            std::cerr << "[Rank 0] [IVF-HNSW Build] Distributed building done.\n";
        }
    }
};