#pragma once
#include <cstdint>
#include <limits>
#include "simd_l2.h"
#include "aligned_alloc.h"
#include "ivfpq_index.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

inline float compute_l2_sqr_d3(const float* a, const float* b) {
    float d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
    return d0 * d0 + d1 * d1 + d2 * d2;
}

struct alignas(64) SearchWorkspace {
    float residual[FS_D];
    float lut_f[FS_M * 16];
    uint8_t lut_u8[FS_M * 16];
    uint8_t query_code[FS_M];
    uint16_t sum_arr[16];
};

__attribute__((always_inline)) inline void pq_build_lut16_d3(
    const float* q, const float* cents16x3, float* lut16) {
    const float q0 = q[0], q1 = q[1], q2 = q[2];
    #pragma GCC unroll 16
    for (int k = 0; k < 16; ++k) {
        const float* c = cents16x3 + k * 3;
        float d0 = q0 - c[0], d1 = q1 - c[1], d2 = q2 - c[2];
        lut16[k] = d0 * d0 + d1 * d1 + d2 * d2;
    }
}

__attribute__((always_inline)) inline uint8_t pq_quantize_d3_16(
    const float* q, const float* cents16x3) {
    float min_dist = std::numeric_limits<float>::max();
    uint8_t best = 0;
    const float q0 = q[0], q1 = q[1], q2 = q[2];
    #pragma GCC unroll 16
    for (int k = 0; k < 16; ++k) {
        const float* c = cents16x3 + k * 3;
        float d0 = q0 - c[0], d1 = q1 - c[1], d2 = q2 - c[2];
        float dist = d0 * d0 + d1 * d1 + d2 * d2;
        if (dist < min_dist) {
            min_dist = dist;
            best = static_cast<uint8_t>(k);
        }
    }
    return best;
}

__attribute__((always_inline)) inline void pq_build_adc_lut(
    const float* residual, int M, int d_sub,
    const float* pq_centroids, float* lut_out,
    float& min_val, float& max_val) {
#if defined(__AVX2__)
    __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256 vmax = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    for (int m = 0; m < M; ++m) {
        const float* sub_q = residual + m * d_sub;
        const float* sub_c = pq_centroids + m * 16 * d_sub;
        float* row = lut_out + m * 16;
        if (d_sub == 3) {
            pq_build_lut16_d3(sub_q, sub_c, row);
        } else {
            for (int k = 0; k < 16; ++k) {
                row[k] = compute_L2_sqr(sub_q, sub_c + k * d_sub, d_sub);
            }
        }
        __m256 r0 = _mm256_loadu_ps(row);
        __m256 r1 = _mm256_loadu_ps(row + 8);
        vmin = _mm256_min_ps(vmin, _mm256_min_ps(r0, r1));
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(r0, r1));
    }
    min_val = hmin_avx2(vmin);
    max_val = hmax_avx2(vmax);
#else
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    for (int m = 0; m < M; ++m) {
        const float* sub_q = residual + m * d_sub;
        const float* sub_c = pq_centroids + m * 16 * d_sub;
        float* row = lut_out + m * 16;
        if (d_sub == 3) {
            pq_build_lut16_d3(sub_q, sub_c, row);
        } else {
            for (int k = 0; k < 16; ++k) {
                row[k] = compute_L2_sqr(sub_q, sub_c + k * d_sub, d_sub);
            }
        }
        for (int k = 0; k < 16; ++k) {
            min_val = min_val < row[k] ? min_val : row[k];
            max_val = max_val > row[k] ? max_val : row[k];
        }
    }
#endif
}

__attribute__((always_inline)) inline void sdc_quantize_and_build_lut(
    const float* residual, int M, int d_sub,
    const float* pq_centroids, const float* center_dist_table,
    float* lut_out, uint8_t* codes_out,
    float& min_val, float& max_val) {
#if defined(__AVX2__)
    __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256 vmax = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    for (int m = 0; m < M; ++m) {
        const float* sub_q = residual + m * d_sub;
        const float* sub_c = pq_centroids + m * 16 * d_sub;
        uint8_t q_code;
        if (d_sub == 3) {
            q_code = pq_quantize_d3_16(sub_q, sub_c);
        } else {
            float min_dist = std::numeric_limits<float>::max();
            uint8_t best = 0;
            for (int k = 0; k < 16; ++k) {
                float dist = compute_L2_sqr(sub_q, sub_c + k * d_sub, d_sub);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = static_cast<uint8_t>(k);
                }
            }
            q_code = best;
        }
        codes_out[m] = q_code;
        const float* row = center_dist_table + m * 256 + q_code * 16;
        float* lut_row = lut_out + m * 16;
        __m256 v0 = _mm256_loadu_ps(row);
        __m256 v1 = _mm256_loadu_ps(row + 8);
        _mm256_storeu_ps(lut_row, v0);
        _mm256_storeu_ps(lut_row + 8, v1);
        vmin = _mm256_min_ps(vmin, _mm256_min_ps(v0, v1));
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(v0, v1));
    }
    min_val = hmin_avx2(vmin);
    max_val = hmax_avx2(vmax);
#else
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    for (int m = 0; m < M; ++m) {
        const float* sub_q = residual + m * d_sub;
        const float* sub_c = pq_centroids + m * 16 * d_sub;
        uint8_t q_code;
        if (d_sub == 3) {
            q_code = pq_quantize_d3_16(sub_q, sub_c);
        } else {
            float min_dist = std::numeric_limits<float>::max();
            uint8_t best = 0;
            for (int k = 0; k < 16; ++k) {
                float dist = compute_L2_sqr(sub_q, sub_c + k * d_sub, d_sub);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = static_cast<uint8_t>(k);
                }
            }
            q_code = best;
        }
        codes_out[m] = q_code;
        const float* row = center_dist_table + m * 256 + q_code * 16;
        float* lut_row = lut_out + m * 16;
        for (int k = 0; k < 16; ++k) {
            lut_row[k] = row[k];
            if (lut_row[k] < min_val) min_val = lut_row[k];
            if (lut_row[k] > max_val) max_val = lut_row[k];
        }
    }
#endif
}

__attribute__((always_inline)) inline void lut_float_to_u8(
    const float* lut, int n, float min_val, float inv_scale, uint8_t* out) {
#if defined(__AVX2__)
    const __m256 vmin = _mm256_set1_ps(min_val);
    const __m256 vscale = _mm256_set1_ps(inv_scale);
    const __m256i vmax_i = _mm256_set1_epi32(255);
    const __m256i vzero = _mm256_setzero_si256();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(lut + i), vmin), vscale);
        __m256i iv = _mm256_cvtps_epi32(v);
        iv = _mm256_max_epi32(vzero, _mm256_min_epi32(iv, vmax_i));
        __m128i lo = _mm256_castsi256_si128(iv);
        __m128i hi = _mm256_extracti128_si256(iv, 1);
        __m128i pack16 = _mm_packus_epi32(lo, hi);
        __m128i pack8 = _mm_packus_epi16(pack16, pack16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i), pack8);
    }
    for (; i < n; ++i) {
        out[i] = static_cast<uint8_t>((lut[i] - min_val) * inv_scale);
    }
#else
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>((lut[i] - min_val) * inv_scale);
    }
#endif
}

#if defined(__ARM_NEON) || defined(__aarch64__)
__attribute__((always_inline)) inline void residual_sub_d96(
    const float* q, const float* c, float* out) {
    for (int i = 0; i < 96; i += 16) {
        float32x4_t vq0 = vld1q_f32(q + i);
        float32x4_t vc0 = vld1q_f32(c + i);
        vst1q_f32(out + i, vsubq_f32(vq0, vc0));
        float32x4_t vq1 = vld1q_f32(q + i + 4);
        float32x4_t vc1 = vld1q_f32(c + i + 4);
        vst1q_f32(out + i + 4, vsubq_f32(vq1, vc1));
        float32x4_t vq2 = vld1q_f32(q + i + 8);
        float32x4_t vc2 = vld1q_f32(c + i + 8);
        vst1q_f32(out + i + 8, vsubq_f32(vq2, vc2));
        float32x4_t vq3 = vld1q_f32(q + i + 12);
        float32x4_t vc3 = vld1q_f32(c + i + 12);
        vst1q_f32(out + i + 12, vsubq_f32(vq3, vc3));
    }
}
#elif defined(__AVX2__)
__attribute__((always_inline)) inline void residual_sub_d96(
    const float* q, const float* c, float* out) {
    for (int i = 0; i < 96; i += 32) {
        __m256 vq0 = _mm256_loadu_ps(q + i);
        __m256 vc0 = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(vq0, vc0));
        __m256 vq1 = _mm256_loadu_ps(q + i + 8);
        __m256 vc1 = _mm256_loadu_ps(c + i + 8);
        _mm256_storeu_ps(out + i + 8, _mm256_sub_ps(vq1, vc1));
        __m256 vq2 = _mm256_loadu_ps(q + i + 16);
        __m256 vc2 = _mm256_loadu_ps(c + i + 16);
        _mm256_storeu_ps(out + i + 16, _mm256_sub_ps(vq2, vc2));
        __m256 vq3 = _mm256_loadu_ps(q + i + 24);
        __m256 vc3 = _mm256_loadu_ps(c + i + 24);
        _mm256_storeu_ps(out + i + 24, _mm256_sub_ps(vq3, vc3));
    }
}
#else
__attribute__((always_inline)) inline void residual_sub_d96(
    const float* q, const float* c, float* out) {
    for (int i = 0; i < 96; ++i) out[i] = q[i] - c[i];
}
#endif
