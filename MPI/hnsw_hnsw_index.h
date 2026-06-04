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

class HNSWHNSWIndex {
public:
    int d;
    int n_lists;
    int hnsw_m;
    int hnsw_ef_construction;

    AlignedVector<float> centroids; 
    
    hnswlib::L2Space* space = nullptr;
    
    hnswlib::HierarchicalNSW<float>* top_hnsw = nullptr;
    
    std::vector<hnswlib::HierarchicalNSW<float>*> bottom_hnsws;

    HNSWHNSWIndex(int dim = 96, int nlist = 1024, int m = 16, int ef_construction = 200) 
        : d(dim), n_lists(nlist), hnsw_m(m), hnsw_ef_construction(ef_construction) {
        
        centroids.resize(n_lists * d);
        bottom_hnsws.resize(n_lists, nullptr);
        
        space = new hnswlib::L2Space(d);
    }

    ~HNSWHNSWIndex() {
        if (top_hnsw != nullptr) {
            delete top_hnsw;
        }
        for (auto* graph : bottom_hnsws) {
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
            std::cerr << "[Rank 0] [HNSW+HNSW Build] d=" << d << ", n_lists=" << n_lists 
                      << ", HNSW_M=" << hnsw_m << ", HNSW_ef_c=" << hnsw_ef_construction << "\n";
        }

        {
            MicroProfiler::Timer _t("1_Train_Centroids_&_Bcast");
            int sample_per_node = std::min(static_cast<size_t>(25000), local_n);
            std::vector<float> local_sample(sample_per_node * d);
            for(int i = 0; i < sample_per_node; ++i) {
                size_t idx = (local_n / sample_per_node) * i; 
                std::memcpy(&local_sample[i * d], &local_data[idx * d], d * sizeof(float));
            }

            std::vector<float> global_sample;
            if (rank == 0) global_sample.resize(sample_per_node * size * d);

            MPI_Gather(local_sample.data(), sample_per_node * d, MPI_FLOAT,
                       global_sample.data(), sample_per_node * d, MPI_FLOAT,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                KMeans km(d, n_lists);
                km.train(global_sample.data(), sample_per_node * size);
                std::memcpy(centroids.data(), km.centroids.data(), n_lists * d * sizeof(float));
            }
            MPI_Bcast(centroids.data(), n_lists * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        {
            MicroProfiler::Timer _t("2_Build_Top_HNSW");
            top_hnsw = new hnswlib::HierarchicalNSW<float>(space, n_lists, hnsw_m, hnsw_ef_construction);
            for (int c = 0; c < n_lists; ++c) {
                // Label为桶的编号 [0, n_lists - 1]
                top_hnsw->addPoint(centroids.data() + c * d, static_cast<hnswlib::labeltype>(c));
            }
        }

        std::vector<int> assign(local_n);
        std::vector<size_t> bucket_counts(n_lists, 0);

        {
            MicroProfiler::Timer _t("3_Assign_Buckets_via_Top_HNSW");
            int num_threads = tp::get_num_threads();
            std::vector<std::vector<size_t>> thread_local_counts(
                static_cast<size_t>(num_threads), std::vector<size_t>(n_lists, 0));

            top_hnsw->setEf(10); 

            tp::parallel_region([&](int tid) {
                const size_t chunk = (local_n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
                const size_t i0 = static_cast<size_t>(tid) * chunk;
                const size_t i1 = std::min(local_n, i0 + chunk);

                for (size_t i = i0; i < i1; ++i) {
                    auto res = top_hnsw->searchKnn(local_data + i * d, 1);
                    int best_c = static_cast<int>(res.top().second);
                    
                    assign[i] = best_c;
                    thread_local_counts[static_cast<size_t>(tid)][best_c]++;
                }
            });

            for (int t = 0; t < num_threads; ++t) {
                for (int c = 0; c < n_lists; ++c) {
                    bucket_counts[c] += thread_local_counts[static_cast<size_t>(t)][c];
                }
            }
        }

        {
            MicroProfiler::Timer _t("4_Init_Bottom_HNSW_Graphs");
            for (int c = 0; c < n_lists; ++c) {
                if (bucket_counts[c] > 0) {
                    bottom_hnsws[c] = new hnswlib::HierarchicalNSW<float>(
                        space, bucket_counts[c], hnsw_m, hnsw_ef_construction);
                }
            }
        }

        {
            MicroProfiler::Timer _t("5_Build_Bottom_HNSW_Graphs");
            tp::parallel_for_static(0, local_n, [&](size_t i) {
                int c = assign[i];
                hnswlib::labeltype global_label = static_cast<hnswlib::labeltype>(rank * local_n + i);
                bottom_hnsws[c]->addPoint((const void*)(local_data + i * d), global_label);
            });
        }
        
        if (rank == 0) {
            std::cerr << "[Rank 0] [HNSW+HNSW Build] Distributed building done.\n";
        }
    }
};