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

constexpr int FS_D = 96;
constexpr int FS_M = 32;
constexpr int FS_K = 16;
constexpr int FS_D_SUB = 3;

struct alignas(16) FSBlock {
    uint8_t codes[FS_M][16];
    uint32_t ids[16];
};

struct InvertedList {
    std::vector<FSBlock> blocks;
    size_t total_elements = 0;
};

struct TempVec {
    uint8_t code[FS_M];
    uint32_t id;
};

class IVFPQIndex {
public:
    int d = FS_D;
    int n_lists;
    int M = FS_M;
    int K = FS_K;
    int d_sub = FS_D_SUB;

    AlignedVector<float> ivf_centroids; 
    AlignedVector<float> pq_centroids;  
    std::vector<InvertedList> lists;  

    IVFPQIndex(int dim = 96, int nlist = 1024) 
        : d(dim), n_lists(nlist) {
        ivf_centroids.resize(n_lists * d);
        pq_centroids.resize(M * K * d_sub);
        lists.resize(n_lists);
    }

    // 传入的是 local_data (当前进程分到的底库切片) 和 MPI 的 rank 与 size
    void build(const float* local_data, size_t local_n, int rank, int size) {
        if (rank == 0) {
            std::cerr << "[Rank 0] [IVFPQ Build] d=" << d << ", n_lists=" << n_lists 
                      << ", M=" << M << ", K=" << K << ", d_sub=" << d_sub << "\n";
        }


        // 全局 IVF 粗聚类中心训练与广播
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

        std::vector<int> assign(local_n);
        std::vector<float> residuals(local_n * d);

        {
            MicroProfiler::Timer _t("2_Compute_Residuals");
            tp::parallel_for_static(0, local_n, [&](size_t i) {
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
                for (int j = 0; j < d; ++j) {
                    residuals[i * d + j] = local_data[i * d + j] - ivf_centroids[best_c * d + j];
                }
            });
        }

        // =========================================================
        // 全局 PQ 量化中心训练与广播
        // =========================================================
        {
            MicroProfiler::Timer _t("3_Train_PQ_&_Bcast");
            if (rank == 0) {
                // Rank 0 利用刚刚算出的本地残差数据，进行 PQ 聚类中心训练
                tp::parallel_for_dynamic(static_cast<size_t>(M), [&](size_t m) {
                    std::vector<float> sub_data(local_n * d_sub);
                    for (size_t i = 0; i < local_n; ++i) {
                        for (int j = 0; j < d_sub; ++j) {
                            sub_data[i * d_sub + j] = residuals[i * d + static_cast<int>(m) * d_sub + j];
                        }
                    }
                    KMeans pq_km(d_sub, K);
                    pq_km.train(sub_data.data(), local_n);
                    std::memcpy(&pq_centroids[static_cast<int>(m) * K * d_sub], pq_km.centroids.data(), K * d_sub * sizeof(float));
                });
            }
            // Master 将 PQ 中心广播给所有 Workers
            MPI_Bcast(pq_centroids.data(), M * K * d_sub, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // =========================================================
        // 各个 Rank 独立对自己负责的局部残差进行 PQ 编码
        // =========================================================
        int num_threads = tp::get_num_threads();
        std::vector<std::vector<std::vector<TempVec>>> thread_local_lists(
            static_cast<size_t>(num_threads), std::vector<std::vector<TempVec>>(n_lists));

        {
            MicroProfiler::Timer _t("4_Encode_PQ");
            tp::parallel_region([&](int tid) {
                const size_t chunk = (local_n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
                const size_t i0 = static_cast<size_t>(tid) * chunk;
                const size_t i1 = std::min(local_n, i0 + chunk);
                for (size_t i = i0; i < i1; ++i) {
                    TempVec tv;
                    // 这里存入倒排链表的 ID 是当前 Worker 的局部相对 ID (i)
                    // 在 main.cc 汇总的时候，会通过 rank_id * local_n + i 还原为全局 ID
                    tv.id = static_cast<uint32_t>(i); 
                    for (int m = 0; m < M; ++m) {
                        const float* sub_res = &residuals[i * d + m * d_sub];
                        const float* sub_cents = &pq_centroids[m * K * d_sub];
                        float min_dist = std::numeric_limits<float>::max();
                        uint8_t best_k = 0;
                        for (int kk = 0; kk < K; ++kk) {
                            float dist = compute_L2_sqr(sub_res, sub_cents + kk * d_sub, d_sub);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_k = static_cast<uint8_t>(kk);
                            }
                        }
                        tv.code[m] = best_k;
                    }
                    thread_local_lists[static_cast<size_t>(tid)][assign[i]].push_back(tv);
                }
            });
        }

        // =========================================================
        // 各个 Rank 并行将本地的 PQ Codes 打包成供 FastScan 使用的结构
        // =========================================================
        {
            MicroProfiler::Timer _t("5_Interleave_Blocks");
            tp::parallel_for_dynamic(static_cast<size_t>(n_lists), [&](size_t c) {
                std::vector<TempVec> merged_list;
                for (int t = 0; t < num_threads; ++t) {
                    merged_list.insert(merged_list.end(),
                        thread_local_lists[static_cast<size_t>(t)][static_cast<int>(c)].begin(),
                        thread_local_lists[static_cast<size_t>(t)][static_cast<int>(c)].end());
                }
                size_t num = merged_list.size();
                lists[static_cast<int>(c)].total_elements = num;
                size_t num_blocks = (num + 15) / 16;
                lists[static_cast<int>(c)].blocks.resize(num_blocks);
                for (size_t b = 0; b < num_blocks; ++b) {
                    FSBlock& block = lists[static_cast<int>(c)].blocks[b];
                    for (int k = 0; k < 16; ++k) {
                        size_t idx = b * 16 + k;
                        if (idx < num) {
                            block.ids[k] = merged_list[idx].id;
                            for (int m = 0; m < M; ++m) {
                                block.codes[m][k] = merged_list[idx].code[m];
                            }
                        } else {
                            block.ids[k] = 0xFFFFFFFF; 
                            for (int m = 0; m < M; ++m) {
                                block.codes[m][k] = 0;
                            }
                        }
                    }
                }
            });
        }
        
        if (rank == 0) {
            std::cerr << "[Rank 0] [IVFPQ Build] Distributed building done.\n";
        }
    }
};