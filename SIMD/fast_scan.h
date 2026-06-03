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

constexpr int FS_D = 96;          // 原始向量维度
constexpr int FS_M = 32;          // 子空间数量32
constexpr int FS_K = 16;          // 聚类中心 16
constexpr int FS_D_SUB = 3;      

struct alignas(64) FSBlock {
    uint8_t codes[FS_M][16];
};

struct alignas(8) Candidate {
    float dist;    
    uint32_t id;
};

// 统一的精确内积距离计算接口
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
    for (size_t d = 0; d < 96; ++d) sum += query_ptr[d] * base_ptr[d];
    return sum;
#endif
}

class FastScanKMeans {
public:
    int d; 
    int k; 
    std::vector<float> centroids; 

    FastScanKMeans(int dim = FS_D_SUB, int num_clusters = FS_K) : d(dim), k(num_clusters) {
        centroids.resize(k * d, 0.0f);
    }

    void train(const float* train_data, size_t n, int max_iter = 20) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());

        for (int c = 0; c < k; ++c) {
            size_t idx = indices[c];
            for (int j = 0; j < d; ++j) centroids[c * d + j] = train_data[idx * d + j];
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
                    // D_SUB = 3 展开优化
                    float diff0 = train_data[i * d + 0] - centroids[c * d + 0];
                    float diff1 = train_data[i * d + 1] - centroids[c * d + 1];
                    float diff2 = train_data[i * d + 2] - centroids[c * d + 2];
                    dist = diff0*diff0 + diff1*diff1 + diff2*diff2;

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assigns[i] = best_c;
                counts[best_c]++;
                
                new_centroids[best_c * d + 0] += train_data[i * d + 0];
                new_centroids[best_c * d + 1] += train_data[i * d + 1];
                new_centroids[best_c * d + 2] += train_data[i * d + 2];
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] == 0) {
                    size_t rand_idx = rand() % n;
                    for (int j = 0; j < d; ++j) centroids[c * d + j] = train_data[rand_idx * d + j];
                } else {
                    for (int j = 0; j < d; ++j) centroids[c * d + j] = new_centroids[c * d + j] / counts[c];
                }
            }
        }
    }
};

class PQQuantizer {
public:
    std::vector<FastScanKMeans> subspaces;

    PQQuantizer() {
        subspaces.resize(FS_M, FastScanKMeans(FS_D_SUB, FS_K));
    }

    void train(const float* base_data, size_t n) {
        std::cerr << "[FastScan Info] Starting KMeans (K=16, M=32) training...\n";
        std::vector<std::vector<float>> sub_train_data(FS_M, std::vector<float>(n * FS_D_SUB));
        
        const int BLOCK_SIZE = 32;

        #pragma omp parallel for schedule(dynamic)
        for (size_t i_blk = 0; i_blk < n; i_blk += BLOCK_SIZE) {
            size_t blk_end = std::min(n, i_blk + BLOCK_SIZE);
            alignas(32) float rotated_vec_blk[BLOCK_SIZE][FS_D] = {0.0f};

            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int j = 0; j < FS_D; ++j) {
                    float val = base_data[i * FS_D + j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t v_val = vdupq_n_f32(val);
                    for (int k = 0; k < FS_D; k += 4) {
                        float32x4_t v_rot = vld1q_f32(&rotated_vec_blk[i - i_blk][k]);
                        float32x4_t v_r = vld1q_f32(&OPQ_R[j][k]);
                        v_rot = vmlaq_f32(v_rot, v_val, v_r);
                        vst1q_f32(&rotated_vec_blk[i - i_blk][k], v_rot);
                    }
#else
                    for (int k = 0; k < FS_D; ++k) rotated_vec_blk[i - i_blk][k] += val * OPQ_R[j][k];
#endif
                }
            }
            
            for (size_t i = i_blk; i < blk_end; ++i) {
                for (int m = 0; m < FS_M; ++m) {
                    for (int d = 0; d < FS_D_SUB; ++d) {
                        sub_train_data[m][i * FS_D_SUB + d] = rotated_vec_blk[i - i_blk][m * FS_D_SUB + d];
                    }
                }
            }
        }
        
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < FS_M; ++m) {
            subspaces[m].train(sub_train_data[m].data(), n);
        }
        std::cerr << "[FastScan Info] KMeans training completed.\n";
    }

    //将数据编码并紧凑写入 FSBlock 转置结构中
    void encode_batch(const float* base_data, void* output_blocks_ptr, size_t n) {
        FSBlock* output_blocks = static_cast<FSBlock*>(output_blocks_ptr);
        size_t num_blocks = n / 32;

        #pragma omp parallel for schedule(dynamic)
        for (size_t blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
            size_t base_offset = blk_idx * 32;
            alignas(32) float rotated_vec_blk[32][FS_D] = {0.0f};
            uint8_t temp_codes[32][FS_M]; 

            // 旋转计算 32 条向量
            for (size_t i = 0; i < 32; ++i) {
                for (int j = 0; j < FS_D; ++j) {
                    float val = base_data[(base_offset + i) * FS_D + j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t v_val = vdupq_n_f32(val);
                    for (int k = 0; k < FS_D; k += 4) {
                        float32x4_t v_rot = vld1q_f32(&rotated_vec_blk[i][k]);
                        float32x4_t v_r = vld1q_f32(&OPQ_R[j][k]);
                        v_rot = vmlaq_f32(v_rot, v_val, v_r);
                        vst1q_f32(&rotated_vec_blk[i][k], v_rot);
                    }
#else
                    for (int k = 0; k < FS_D; ++k) rotated_vec_blk[i][k] += val * OPQ_R[j][k];
#endif
                }
            }

            // 聚类求 ID
            for (size_t i = 0; i < 32; ++i) {
                for (int m = 0; m < FS_M; ++m) {
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;
                    const float* sub_vec = rotated_vec_blk[i] + m * FS_D_SUB;
                    const float* centroids = subspaces[m].centroids.data();

                    for (int c = 0; c < FS_K; ++c) {
                        float diff0 = sub_vec[0] - centroids[c * FS_D_SUB + 0];
                        float diff1 = sub_vec[1] - centroids[c * FS_D_SUB + 1];
                        float diff2 = sub_vec[2] - centroids[c * FS_D_SUB + 2];
                        float dist = diff0*diff0 + diff1*diff1 + diff2*diff2;

                        if (dist < min_dist) {
                            min_dist = dist;
                            best_c = c;
                        }
                    }
                    temp_codes[i][m] = static_cast<uint8_t>(best_c);
                }
            }


            std::memset(&output_blocks[blk_idx], 0, sizeof(FSBlock));
            for (int m = 0; m < FS_M; ++m) {
                for (int v = 0; v < 32; v += 2) {
                    // 每两个向量合并进 1 byte：低 4 bit 存偶数向量ID，高 4 bit 存奇数向量ID
                    uint8_t code_even = temp_codes[v][m] & 0x0F;
                    uint8_t code_odd  = temp_codes[v+1][m] & 0x0F;
                    output_blocks[blk_idx].codes[m][v / 2] = code_even | (code_odd << 4);
                }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const void* base_blocks_ptr, 
        const float* original_base, 
        const float* query, 
        size_t base_number, 
        size_t top_k,
        int top_c 
    ) {
        const FSBlock* base_blocks = static_cast<const FSBlock*>(base_blocks_ptr);
        size_t num_blocks = base_number / 32;

        alignas(32) float rotated_query[FS_D] = {0.0f};
        {
            MicroProfiler::Timer t("1_Rotate_Query");
            for (int j = 0; j < FS_D; ++j) {
                float q_val = query[j];
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t v_q = vdupq_n_f32(q_val);
                for (int i = 0; i < FS_D; i += 4) {
                    float32x4_t v_rot = vld1q_f32(&rotated_query[i]);
                    float32x4_t v_r = vld1q_f32(&OPQ_R[j][i]);
                    v_rot = vmlaq_f32(v_rot, v_q, v_r);
                    vst1q_f32(&rotated_query[i], v_rot);
                }
#else
                for (int i = 0; i < FS_D; ++i) rotated_query[i] += q_val * OPQ_R[j][i];
#endif
            }
        }

        // 包含量化过程的 LUT 构建
        alignas(64) uint8_t lut_u8[FS_M][16] = {0}; 
        {
            MicroProfiler::Timer t("2_Build_LUT");
            float min_ip = std::numeric_limits<float>::max();
            float max_ip = -std::numeric_limits<float>::max();
            float lut_float[FS_M][FS_K];

            for (int m = 0; m < FS_M; ++m) {
                const float* sub_query = rotated_query + m * FS_D_SUB;
                const float* centroids = subspaces[m].centroids.data();
                for (int c = 0; c < FS_K; ++c) {
                    float ip = sub_query[0]*centroids[c*FS_D_SUB+0] + 
                               sub_query[1]*centroids[c*FS_D_SUB+1] + 
                               sub_query[2]*centroids[c*FS_D_SUB+2];
                    lut_float[m][c] = ip;
                    if (ip > max_ip) max_ip = ip;
                    if (ip < min_ip) min_ip = ip;
                }
            }

            float scale = 255.0f / (max_ip - min_ip + 1e-6f);
            for (int m = 0; m < FS_M; ++m) {
                for (int c = 0; c < FS_K; ++c) {
                    lut_u8[m][c] = static_cast<uint8_t>((max_ip - lut_float[m][c]) * scale);
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
                
                uint16_t local_threshold = std::numeric_limits<uint16_t>::max();
                
                #pragma omp for schedule(dynamic, 64)
                for (size_t blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
#if defined(__ARM_NEON) || defined(__aarch64__)
                    // 定义 4 个 16-bit 累加器，涵盖 32 条向量的距离
                    uint16x8_t sum_lo = vdupq_n_u16(0); // Vector 0-7
                    uint16x8_t sum_ml = vdupq_n_u16(0); // Vector 8-15
                    uint16x8_t sum_mh = vdupq_n_u16(0); // Vector 16-23
                    uint16x8_t sum_hi = vdupq_n_u16(0); // Vector 24-31

                    uint8x16_t mask_0f = vdupq_n_u8(0x0F);

                    // 核心查表流水线
                    #pragma GCC unroll 4
                    for (int m = 0; m < FS_M; ++m) {
                        // 加载当前子空间量化好的 16位 LUT 字典
                        uint8x16_t lut = vld1q_u8(lut_u8[m]);
                        // 加载打包好的 32个向量的压缩索引 (16 bytes)
                        uint8x16_t packed = vld1q_u8(base_blocks[blk_idx].codes[m]);

                        // 位运算解包
                        uint8x16_t idx_even = vandq_u8(packed, mask_0f);
                        uint8x16_t idx_odd  = vshrq_n_u8(packed, 4);

                        uint8x16_t dist_even = vqtbl1q_u8(lut, idx_even);
                        uint8x16_t dist_odd  = vqtbl1q_u8(lut, idx_odd);

                        // 并行累加
                        sum_lo = vaddw_u8(sum_lo, vget_low_u8(dist_even));
                        sum_ml = vaddw_u8(sum_ml, vget_high_u8(dist_even));
                        sum_mh = vaddw_u8(sum_mh, vget_low_u8(dist_odd));
                        sum_hi = vaddw_u8(sum_hi, vget_high_u8(dist_odd));
                    }

                    // 提取结果并维护 Top-C
                    alignas(32) uint16_t dist_arr[32];
                    vst1q_u16(dist_arr + 0,  sum_lo);
                    vst1q_u16(dist_arr + 8,  sum_ml);
                    vst1q_u16(dist_arr + 16, sum_mh);
                    vst1q_u16(dist_arr + 24, sum_hi);

                    uint32_t base_id = blk_idx * 32;
                    for(int i=0; i<8; ++i) { 
                        uint16_t d_0 = dist_arr[i];        // even 0-7
                        uint16_t d_1 = dist_arr[i + 16];   // odd 0-7
                        uint16_t d_2 = dist_arr[i + 8];    // even 8-15
                        uint16_t d_3 = dist_arr[i + 24];   // odd 8-15

                        if (d_0 < local_threshold) local_buffer.push_back({(float)d_0, base_id + i * 2});
                        if (d_1 < local_threshold) local_buffer.push_back({(float)d_1, base_id + i * 2 + 1});
                        if (d_2 < local_threshold) local_buffer.push_back({(float)d_2, base_id + i * 2 + 16});
                        if (d_3 < local_threshold) local_buffer.push_back({(float)d_3, base_id + i * 2 + 17});
                    }
#endif
                    if (local_buffer.size() >= buffer_capacity) {
                        std::nth_element(local_buffer.begin(), local_buffer.begin() + top_c, local_buffer.end(),
                                         [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
                        local_buffer.resize(top_c);
                        local_threshold = static_cast<uint16_t>(local_buffer.back().dist);
                    }
                }

                if (local_buffer.size() > top_c) {
                    std::nth_element(local_buffer.begin(), local_buffer.begin() + top_c, local_buffer.end(),
                                     [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
                    local_buffer.resize(top_c);
                }

                #pragma omp critical
                {
                    global_buffer.insert(global_buffer.end(), local_buffer.begin(), local_buffer.end());
                }
            }

            if (global_buffer.size() > top_c) {
                std::nth_element(global_buffer.begin(), global_buffer.begin() + top_c, global_buffer.end(),
                                 [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
                global_buffer.resize(top_c);
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> final_pq;

        {
            MicroProfiler::Timer t("4_Exact_Rerank");
            std::sort(global_buffer.begin(), global_buffer.end(), [](const Candidate& a, const Candidate& b) { return a.id < b.id; });

            const int RERANK_PREFETCH_DIST = 4;
            for (size_t i = 0; i < global_buffer.size(); ++i) {
                if (i + RERANK_PREFETCH_DIST < global_buffer.size()) {
                    uint32_t next_id = global_buffer[i + RERANK_PREFETCH_DIST].id;
                    __builtin_prefetch(original_base + next_id * FS_D, 0, 0);
                }
                uint32_t id = global_buffer[i].id;
                float exact_ip = compute_IP_distance(query, original_base + id * FS_D);
                global_buffer[i].dist = 1.0f - exact_ip; 
            }

            size_t actual_k = std::min(top_k, global_buffer.size());
            std::partial_sort(global_buffer.begin(), global_buffer.begin() + actual_k, global_buffer.end(),
                              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

            for (size_t i = 0; i < actual_k; ++i) final_pq.push({global_buffer[i].dist, global_buffer[i].id});
        }
        return final_pq;
    }
};