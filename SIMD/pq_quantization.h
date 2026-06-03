#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cstring>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "opq_matrix.h" 
#include "profiler.h"

constexpr int PQ_D = 96;          // 原始向量维度
constexpr int PQ_M = 16;          // 子空间数量16
constexpr int PQ_K = 256;         // 每个子空间的聚类中心数
constexpr int PQ_D_SUB = 6;       // 每个子空间的维度

struct alignas(16) PQCode {
    uint8_t code[PQ_M];
};

struct alignas(8) Candidate {
    float dist;
    uint32_t id;
};

// 统一的距离计算接口声明
inline float compute_IP_distance(const float* query_ptr, const float* base_ptr) __attribute__((always_inline));

inline float compute_IP_distance(const float* query_ptr, const float* base_ptr) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < 96; d += 16) {
        float32x4_t q0 = vld1q_f32(query_ptr + d);
        float32x4_t b0 = vld1q_f32(base_ptr + d);
        sum0 = vmlaq_f32(sum0, q0, b0);

        float32x4_t q1 = vld1q_f32(query_ptr + d + 4);
        float32x4_t b1 = vld1q_f32(base_ptr + d + 4);
        sum1 = vmlaq_f32(sum1, q1, b1);

        float32x4_t q2 = vld1q_f32(query_ptr + d + 8);
        float32x4_t b2 = vld1q_f32(base_ptr + d + 8);
        sum2 = vmlaq_f32(sum2, q2, b2);

        float32x4_t q3 = vld1q_f32(query_ptr + d + 12);
        float32x4_t b3 = vld1q_f32(base_ptr + d + 12);
        sum3 = vmlaq_f32(sum3, q3, b3);
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    float sum_arr[4];
    vst1q_f32(sum_arr, sum0);
    return sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

#else
    float sum = 0.0f;
    for (size_t d = 0; d < 96; ++d) {
        sum += query_ptr[d] * base_ptr[d];
    }
    return sum;
#endif
}

class SubspaceKMeans {
public:
    int d; 
    int k; 
    std::vector<float> centroids; 

    SubspaceKMeans(int dim = PQ_D_SUB, int num_clusters = PQ_K) : d(dim), k(num_clusters) {
        centroids.resize(k * d, 0.0f);
    }

    void train(const float* train_data, size_t n, int max_iter = 20) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());

        for (int c = 0; c < k; ++c) {
            size_t idx = indices[c];
            for (int j = 0; j < d; ++j) {
                centroids[c * d + j] = train_data[idx * d + j];
            }
        }

        std::vector<int> assigns(n, 0);
        std::vector<float> new_centroids(k * d, 0.0f);
        std::vector<int> counts(k, 0);

        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < k; ++c) {
                    float dist = 0.0f;
                    if (d == 6) {
                        float d0 = train_data[i * d + 0] - centroids[c * d + 0];
                        float d1 = train_data[i * d + 1] - centroids[c * d + 1];
                        float d2 = train_data[i * d + 2] - centroids[c * d + 2];
                        float d3 = train_data[i * d + 3] - centroids[c * d + 3];
                        float d4 = train_data[i * d + 4] - centroids[c * d + 4];
                        float d5 = train_data[i * d + 5] - centroids[c * d + 5];
                        dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
                    } else {
                        for (int j = 0; j < d; ++j) {
                            float diff = train_data[i * d + j] - centroids[c * d + j];
                            dist += diff * diff;
                        }
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assigns[i] = best_c;
                counts[best_c]++;
                
                if (d == 6) {
                    new_centroids[best_c * d + 0] += train_data[i * d + 0];
                    new_centroids[best_c * d + 1] += train_data[i * d + 1];
                    new_centroids[best_c * d + 2] += train_data[i * d + 2];
                    new_centroids[best_c * d + 3] += train_data[i * d + 3];
                    new_centroids[best_c * d + 4] += train_data[i * d + 4];
                    new_centroids[best_c * d + 5] += train_data[i * d + 5];
                } else {
                    for (int j = 0; j < d; ++j) {
                        new_centroids[best_c * d + j] += train_data[i * d + j];
                    }
                }
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] == 0) {
                    size_t rand_idx = rand() % n;
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = train_data[rand_idx * d + j];
                    }
                } else {
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = new_centroids[c * d + j] / counts[c];
                    }
                }
            }
        }
    }
};

class PQQuantizer {
public:
    std::vector<SubspaceKMeans> subspaces;

    PQQuantizer() {
        subspaces.resize(PQ_M, SubspaceKMeans(PQ_D_SUB, PQ_K));
    }

    void train(const float* base_data, size_t n) {
        std::cerr << "[PQ Info] Starting KMeans training for " << PQ_M << " subspaces...\n";
        std::vector<std::vector<float>> sub_train_data(PQ_M, std::vector<float>(n * PQ_D_SUB));
        
        const int BLOCK_SIZE = 32;

        #pragma omp parallel for schedule(dynamic)
        for (size_t i_blk = 0; i_blk < n; i_blk += BLOCK_SIZE) {
            size_t blk_end = std::min(n, i_blk + BLOCK_SIZE);
            alignas(32) float rotated_vec_blk[BLOCK_SIZE][PQ_D] = {0.0f};

            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int j = 0; j < PQ_D; ++j) {
                    float val = base_data[i * PQ_D + j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t v_val = vdupq_n_f32(val);
                    for (int k = 0; k < PQ_D; k += 4) {
                        float32x4_t v_rot = vld1q_f32(&rotated_vec_blk[i - i_blk][k]);
                        float32x4_t v_r = vld1q_f32(&OPQ_R[j][k]);
                        v_rot = vmlaq_f32(v_rot, v_val, v_r);
                        vst1q_f32(&rotated_vec_blk[i - i_blk][k], v_rot);
                    }
#else
                    for (int k = 0; k < PQ_D; ++k) {
                        rotated_vec_blk[i - i_blk][k] += val * OPQ_R[j][k];
                    }
#endif
                }
            }
            
            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int m = 0; m < PQ_M; ++m) {
                    for (int d = 0; d < PQ_D_SUB; ++d) {
                        sub_train_data[m][i * PQ_D_SUB + d] = rotated_vec_blk[i - i_blk][m * PQ_D_SUB + d];
                    }
                }
            }
        }
        
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < PQ_M; ++m) {
            subspaces[m].train(sub_train_data[m].data(), n);
        }
        std::cerr << "[PQ Info] KMeans training completed.\n";
    }

    void encode_batch(const float* base_data, PQCode* output_codes, size_t n) {
        const int BLOCK_SIZE = 32;

        #pragma omp parallel for schedule(dynamic)
        for (size_t i_blk = 0; i_blk < n; i_blk += BLOCK_SIZE) {
            size_t blk_end = std::min(n, i_blk + BLOCK_SIZE);
            alignas(32) float rotated_vec_blk[BLOCK_SIZE][PQ_D] = {0.0f};

            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int j = 0; j < PQ_D; ++j) {
                    float val = base_data[i * PQ_D + j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t v_val = vdupq_n_f32(val);
                    for (int k = 0; k < PQ_D; k += 4) {
                        float32x4_t v_rot = vld1q_f32(&rotated_vec_blk[i - i_blk][k]);
                        float32x4_t v_r = vld1q_f32(&OPQ_R[j][k]);
                        v_rot = vmlaq_f32(v_rot, v_val, v_r);
                        vst1q_f32(&rotated_vec_blk[i - i_blk][k], v_rot);
                    }
#else
                    for (int k = 0; k < PQ_D; ++k) {
                        rotated_vec_blk[i - i_blk][k] += val * OPQ_R[j][k];
                    }
#endif
                }
            }

            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int m = 0; m < PQ_M; ++m) {
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;
                    
                    const float* sub_vec = rotated_vec_blk[i - i_blk] + m * PQ_D_SUB;
                    const float* centroids = subspaces[m].centroids.data();

                    for (int c = 0; c < PQ_K; ++c) {
                        float dist = 0.0f;
                        if (PQ_D_SUB == 6) {
                            float d0 = sub_vec[0] - centroids[c * PQ_D_SUB + 0];
                            float d1 = sub_vec[1] - centroids[c * PQ_D_SUB + 1];
                            float d2 = sub_vec[2] - centroids[c * PQ_D_SUB + 2];
                            float d3 = sub_vec[3] - centroids[c * PQ_D_SUB + 3];
                            float d4 = sub_vec[4] - centroids[c * PQ_D_SUB + 4];
                            float d5 = sub_vec[5] - centroids[c * PQ_D_SUB + 5];
                            dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
                        } else {
                            for (int j = 0; j < PQ_D_SUB; ++j) {
                                float diff = sub_vec[j] - centroids[c * PQ_D_SUB + j];
                                dist += diff * diff;
                            }
                        }

                        if (dist < min_dist) {
                            min_dist = dist;
                            best_c = c;
                        }
                    }
                    output_codes[i].code[m] = static_cast<uint8_t>(best_c);
                }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const PQCode* base_codes, 
        const float* original_base, 
        const float* query, 
        size_t base_number, 
        size_t top_k,
        int top_c 
    ) {
        alignas(32) float rotated_query[PQ_D] = {0.0f};
        {
            MicroProfiler::Timer t("1_Rotate_Query");
            for (int j = 0; j < PQ_D; ++j) {
                float q_val = query[j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t v_q = vdupq_n_f32(q_val);
                for (int i = 0; i < PQ_D; i += 4) {
                    float32x4_t v_rot = vld1q_f32(&rotated_query[i]);
                    float32x4_t v_r = vld1q_f32(&OPQ_R[j][i]);
                    v_rot = vmlaq_f32(v_rot, v_q, v_r);
                    vst1q_f32(&rotated_query[i], v_rot);
                }
#else
                for (int i = 0; i < PQ_D; ++i) {
                    rotated_query[i] += q_val * OPQ_R[j][i];
                }
#endif
            }
        }

        alignas(32) float lut[PQ_M][PQ_K]; 
        {
            MicroProfiler::Timer t("2_Build_LUT");
            for (int m = 0; m < PQ_M; ++m) {
                const float* sub_query = rotated_query + m * PQ_D_SUB;
                const float* centroids = subspaces[m].centroids.data();
                for (int c = 0; c < PQ_K; ++c) {
                    float ip = 0.0f;
                    for (int j = 0; j < PQ_D_SUB; ++j) {
                        ip += sub_query[j] * centroids[c * PQ_D_SUB + j];
                    }
                    lut[m][c] = ip;
                }
            }
        }

        std::vector<Candidate> global_buffer;
        global_buffer.reserve(omp_get_max_threads() * top_c * 2);

        {
            MicroProfiler::Timer t("3_ADC_Scan");
            #pragma omp parallel
            {
                const int buffer_capacity = std::max(2048, top_c * 8);
                std::vector<Candidate> local_buffer;
                local_buffer.reserve(buffer_capacity);
                
                float local_threshold = std::numeric_limits<float>::max();
                
                #pragma omp for schedule(static)
                for (size_t i = 0; i < base_number; ++i) {

                    float total_ip = 0.0f;
                    const uint8_t* code = base_codes[i].code;

                    #pragma GCC unroll 16
                    for (int m = 0; m < PQ_M; ++m) {
                        total_ip += lut[m][code[m]];
                    }

                    float final_dist = 1.0f - total_ip;

                    if (final_dist < local_threshold) {
                        local_buffer.push_back({final_dist, static_cast<uint32_t>(i)});
                        
                        if (local_buffer.size() == buffer_capacity) {
                            std::nth_element(local_buffer.begin(), 
                                             local_buffer.begin() + top_c, 
                                             local_buffer.end(),
                                             [](const Candidate& a, const Candidate& b) {
                                                 return a.dist < b.dist;
                                             });
                            local_buffer.resize(top_c);
                            local_threshold = local_buffer.back().dist; 
                        }
                    }
                }

                if (local_buffer.size() > top_c) {
                    std::nth_element(local_buffer.begin(), 
                                     local_buffer.begin() + top_c, 
                                     local_buffer.end(),
                                     [](const Candidate& a, const Candidate& b) {
                                         return a.dist < b.dist;
                                     });
                    local_buffer.resize(top_c);
                }

                #pragma omp critical
                {
                    global_buffer.insert(global_buffer.end(), local_buffer.begin(), local_buffer.end());
                }
            }

            if (global_buffer.size() > top_c) {
                std::nth_element(global_buffer.begin(), 
                                 global_buffer.begin() + top_c, 
                                 global_buffer.end(),
                                 [](const Candidate& a, const Candidate& b) {
                                     return a.dist < b.dist;
                                 });
                global_buffer.resize(top_c);
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> final_pq;

        {
            MicroProfiler::Timer t("4_Exact_Rerank");
            std::sort(global_buffer.begin(), global_buffer.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.id < b.id;
                      });

            const int RERANK_PREFETCH_DIST = 4;
            for (size_t i = 0; i < global_buffer.size(); ++i) {
                if (i + RERANK_PREFETCH_DIST < global_buffer.size()) {
                    uint32_t next_id = global_buffer[i + RERANK_PREFETCH_DIST].id;
                    __builtin_prefetch(original_base + next_id * PQ_D, 0, 0);
                }
                
                uint32_t id = global_buffer[i].id;
                
                float exact_ip = compute_IP_distance(query, original_base + id * PQ_D);
                global_buffer[i].dist = 1.0f - exact_ip; 
            }

            size_t actual_k = std::min(top_k, global_buffer.size());
            std::partial_sort(global_buffer.begin(), 
                              global_buffer.begin() + actual_k, 
                              global_buffer.end(),
                              [](const Candidate& a, const Candidate& b) {
                                  return a.dist < b.dist;
                              });

            for (size_t i = 0; i < actual_k; ++i) {
                final_pq.push({global_buffer[i].dist, global_buffer[i].id});
            }
        }

        return final_pq;
    }
};