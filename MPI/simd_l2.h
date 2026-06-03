#pragma once
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow  = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, 1);
    vlow  = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

inline float hmin_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(lo);
}

inline float hmax_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(lo);
}
#endif

inline float compute_L2_sqr(const float* a, const float* b, int d) __attribute__((always_inline));
inline float compute_L2_sqr(const float* a, const float* b, int d) {
    if (d == 96) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);
        for (int i = 0; i < 96; i += 16) {
            float32x4_t a0 = vld1q_f32(a + i), b0 = vld1q_f32(b + i);
            float32x4_t a1 = vld1q_f32(a + i + 4), b1 = vld1q_f32(b + i + 4);
            float32x4_t a2 = vld1q_f32(a + i + 8), b2 = vld1q_f32(b + i + 8);
            float32x4_t a3 = vld1q_f32(a + i + 12), b3 = vld1q_f32(b + i + 12);
            
            float32x4_t sub0 = vsubq_f32(a0, b0); sum0 = vfmaq_f32(sum0, sub0, sub0);
            float32x4_t sub1 = vsubq_f32(a1, b1); sum1 = vfmaq_f32(sum1, sub1, sub1);
            float32x4_t sub2 = vsubq_f32(a2, b2); sum2 = vfmaq_f32(sum2, sub2, sub2);
            float32x4_t sub3 = vsubq_f32(a3, b3); sum3 = vfmaq_f32(sum3, sub3, sub3);
        }
        sum0 = vaddq_f32(sum0, sum1); 
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        return vaddvq_f32(sum0);

#elif defined(__AVX2__)
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int i = 0; i < 96; i += 16) {
            // 使用非对齐加载保证安全性，在支持AVX2的CPU上对齐惩罚极小
            __m256 a0 = _mm256_loadu_ps(a + i), b0 = _mm256_loadu_ps(b + i);
            __m256 a1 = _mm256_loadu_ps(a + i + 8), b1 = _mm256_loadu_ps(b + i + 8);
            __m256 sub0 = _mm256_sub_ps(a0, b0); sum0 = _mm256_fmadd_ps(sub0, sub0, sum0);
            __m256 sub1 = _mm256_sub_ps(a1, b1); sum1 = _mm256_fmadd_ps(sub1, sub1, sum1);
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        return hsum_avx2(sum0);

#else
        float dist = 0;
        for (int i = 0; i < 96; ++i) {
            float diff = a[i] - b[i]; dist += diff * diff;
        }
        return dist;
#endif
    } else {
        // d_sub = 3 时回退标量计算
        float dist = 0;
        for (int i = 0; i < d; ++i) {
            float diff = a[i] - b[i]; dist += diff * diff;
        }
        return dist;
    }
}

inline void compute_all_L2_sqr_d96(const float* q, const float* cents, int nlist, float* out) {
    int c = 0;
    for (; c + 4 <= nlist; c += 4) {
        out[c]     = compute_L2_sqr(q, cents + c * 96, 96);
        out[c + 1] = compute_L2_sqr(q, cents + (c + 1) * 96, 96);
        out[c + 2] = compute_L2_sqr(q, cents + (c + 2) * 96, 96);
        out[c + 3] = compute_L2_sqr(q, cents + (c + 3) * 96, 96);
    }
    for (; c < nlist; ++c) {
        out[c] = compute_L2_sqr(q, cents + c * 96, 96);
    }
}

inline void rerank_batch_d96(const float* q, const float* base, int d,
                             uint32_t* ids, float* dists, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        dists[i]     = compute_L2_sqr(q, base + ids[i] * d, d);
        dists[i + 1] = compute_L2_sqr(q, base + ids[i + 1] * d, d);
        dists[i + 2] = compute_L2_sqr(q, base + ids[i + 2] * d, d);
        dists[i + 3] = compute_L2_sqr(q, base + ids[i + 3] * d, d);
    }
    for (; i < n; ++i) {
        dists[i] = compute_L2_sqr(q, base + ids[i] * d, d);
    }
}