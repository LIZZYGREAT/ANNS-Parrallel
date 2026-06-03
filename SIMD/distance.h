#pragma once
#include <arm_neon.h>
#include <cstddef>

// 强制内联，消除函数调用开销
inline float32x4_t compute_L2_distance_neon_aosoa(
    const float* query_ptr,
    const float* base_block_ptr,
    const size_t dim
) __attribute__((always_inline));

inline float32x4_t compute_L2_distance_neon_aosoa(
    const float* query_ptr,
    const float* base_block_ptr,
    const size_t dim
) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 4) {
        // --- 维度 d ---
        float32x4_t q0 = vld1q_dup_f32(query_ptr + d);
        float32x4_t b0 = vld1q_f32(base_block_ptr + d * 4);
        float32x4_t diff0 = vsubq_f32(q0, b0);
        sum0 = vmlaq_f32(sum0, diff0, diff0);

        // --- 维度 d + 1 ---
        float32x4_t q1 = vld1q_dup_f32(query_ptr + d + 1);
        float32x4_t b1 = vld1q_f32(base_block_ptr + (d + 1) * 4);
        float32x4_t diff1 = vsubq_f32(q1, b1);
        sum1 = vmlaq_f32(sum1, diff1, diff1);

        // --- 维度 d + 2 ---
        float32x4_t q2 = vld1q_dup_f32(query_ptr + d + 2);
        float32x4_t b2 = vld1q_f32(base_block_ptr + (d + 2) * 4);
        float32x4_t diff2 = vsubq_f32(q2, b2);
        sum2 = vmlaq_f32(sum2, diff2, diff2);

        // --- 维度 d + 3 ---
        float32x4_t q3 = vld1q_dup_f32(query_ptr + d + 3);
        float32x4_t b3 = vld1q_f32(base_block_ptr + (d + 3) * 4);
        float32x4_t diff3 = vsubq_f32(q3, b3);
        sum3 = vmlaq_f32(sum3, diff3, diff3);
    }

    // 3. 垂直合并 4 个独立的累加器
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    return vaddq_f32(sum0, sum2);
}