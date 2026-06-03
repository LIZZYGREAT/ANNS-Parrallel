#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <new>
#include <mpi.h> 

#include "profiler.h"
#include "simd_l2.h"
#include "aligned_alloc.h"
#include "thread_pool.h"
#include "hnswlib/hnswlib.h"

class HNSWIndex {
public:
    int d;
    int hnsw_m;
    int hnsw_ef_construction;

    hnswlib::L2Space* space = nullptr;
    hnswlib::HierarchicalNSW<float>* hnsw_graph = nullptr;

    HNSWIndex(int dim = 96, int m = 16, int ef_construction = 200) 
        : d(dim), hnsw_m(m), hnsw_ef_construction(ef_construction) {
        
        space = new hnswlib::L2Space(d);
    }

    ~HNSWIndex() {
        if (hnsw_graph != nullptr) {
            delete hnsw_graph;
        }
        if (space != nullptr) {
            delete space;
        }
    }

    void build(const float* local_data, size_t local_n, int rank, int size) {
        if (rank == 0) {
            std::cerr << "[Rank 0] [Pure HNSW Build] d=" << d 
                      << ", HNSW_M=" << hnsw_m << ", HNSW_ef_c=" << hnsw_ef_construction << "\n";
        }

        {
            MicroProfiler::Timer _t("1_Init_HNSW_Graph");
            hnsw_graph = new hnswlib::HierarchicalNSW<float>(
                space, local_n, hnsw_m, hnsw_ef_construction);
        }

        {
            MicroProfiler::Timer _t("2_Build_HNSW_Graph");
            
            tp::parallel_for_static(0, local_n, [&](size_t i) {
                hnswlib::labeltype global_label = static_cast<hnswlib::labeltype>(rank * local_n + i);
                
                hnsw_graph->addPoint(
                    reinterpret_cast<const void*>(local_data + i * d), 
                    global_label
                );
            });
        }
        
        if (rank == 0) {
            std::cerr << "[Rank 0] [Pure HNSW Build] Distributed sharded building done.\n";
        }
    }
};