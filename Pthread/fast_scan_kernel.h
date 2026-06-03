#pragma once
#include <cstdint>
#include <cstddef>
#include "ivfpq_index.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block, const uint8_t* lut_u8, int M, uint16_t sum_out[16]) {
    uint16x8_t p0_lo = vdupq_n_u16(0), p0_hi = vdupq_n_u16(0);
    uint16x8_t p1_lo = vdupq_n_u16(0), p1_hi = vdupq_n_u16(0);
    uint16x8_t p2_lo = vdupq_n_u16(0), p2_hi = vdupq_n_u16(0);
    uint16x8_t p3_lo = vdupq_n_u16(0), p3_hi = vdupq_n_u16(0);

    #define FS_ACCUM_M(mi, plo, phi) do { \
        uint8x16_t c = vld1q_u8(block.codes[mi]); \
        uint8x16_t l = vld1q_u8(lut_u8 + (mi) * 16); \
        uint8x16_t v = vqtbl1q_u8(l, c); \
        plo = vaddw_u8(plo, vget_low_u8(v)); \
        phi = vaddw_u8(phi, vget_high_u8(v)); \
    } while (0)

    int m = 0;
    for (; m + 4 <= M; m += 4) {
        FS_ACCUM_M(m,     p0_lo, p0_hi);
        FS_ACCUM_M(m + 1, p1_lo, p1_hi);
        FS_ACCUM_M(m + 2, p2_lo, p2_hi);
        FS_ACCUM_M(m + 3, p3_lo, p3_hi);
    }
    for (; m < M; ++m) {
        FS_ACCUM_M(m, p0_lo, p0_hi);
    }
    #undef FS_ACCUM_M

    uint16x8_t sum_lo = vaddq_u16(vaddq_u16(p0_lo, p1_lo), vaddq_u16(p2_lo, p3_lo));
    uint16x8_t sum_hi = vaddq_u16(vaddq_u16(p0_hi, p1_hi), vaddq_u16(p2_hi, p3_hi));
    vst1q_u16(sum_out, sum_lo);
    vst1q_u16(sum_out + 8, sum_hi);
}

#elif defined(__AVX2__)
#include <immintrin.h>

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block, const uint8_t* lut_u8, int M, uint16_t sum_out[16]) {
    __m256i p0 = _mm256_setzero_si256();
    __m256i p1 = _mm256_setzero_si256();
    __m256i p2 = _mm256_setzero_si256();
    __m256i p3 = _mm256_setzero_si256();
    __m256i p4 = _mm256_setzero_si256();
    __m256i p5 = _mm256_setzero_si256();
    __m256i p6 = _mm256_setzero_si256();
    __m256i p7 = _mm256_setzero_si256();

    #define FS_ACCUM_M_AVX(mi, acc) do { \
        __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[mi])); \
        __m128i l = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + (mi) * 16)); \
        acc = _mm256_add_epi16(acc, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l, c))); \
    } while (0)

    int m = 0;
    for (; m + 8 <= M; m += 8) {
        FS_ACCUM_M_AVX(m,     p0);
        FS_ACCUM_M_AVX(m + 1, p1);
        FS_ACCUM_M_AVX(m + 2, p2);
        FS_ACCUM_M_AVX(m + 3, p3);
        FS_ACCUM_M_AVX(m + 4, p4);
        FS_ACCUM_M_AVX(m + 5, p5);
        FS_ACCUM_M_AVX(m + 6, p6);
        FS_ACCUM_M_AVX(m + 7, p7);
    }
    for (; m < M; ++m) {
        FS_ACCUM_M_AVX(m, p0);
    }
    #undef FS_ACCUM_M_AVX

    __m256i sum = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3)),
        _mm256_add_epi16(_mm256_add_epi16(p4, p5), _mm256_add_epi16(p6, p7)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(sum_out), sum);
}

#else

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block, const uint8_t* lut_u8, int M, uint16_t sum_out[16]) {
    for (int k = 0; k < 16; ++k) sum_out[k] = 0;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < 16; ++k) {
            sum_out[k] += lut_u8[m * 16 + block.codes[m][k]];
        }
    }
}

#endif

template<typename ScoreFn>
__attribute__((always_inline)) inline void fast_scan_list_batch(
    const FSBlock* blocks, size_t nb,
    const uint8_t* lut_u8, int M,
    uint16_t* sum_scratch, ScoreFn&& score_block) {
    size_t b = 0;
    for (; b + 4 <= nb; b += 4) {
        fast_scan_block_accumulate(blocks[b],     lut_u8, M, sum_scratch);
        score_block(blocks[b],     sum_scratch);
        fast_scan_block_accumulate(blocks[b + 1], lut_u8, M, sum_scratch);
        score_block(blocks[b + 1], sum_scratch);
        fast_scan_block_accumulate(blocks[b + 2], lut_u8, M, sum_scratch);
        score_block(blocks[b + 2], sum_scratch);
        fast_scan_block_accumulate(blocks[b + 3], lut_u8, M, sum_scratch);
        score_block(blocks[b + 3], sum_scratch);
    }
    for (; b < nb; ++b) {
        fast_scan_block_accumulate(blocks[b], lut_u8, M, sum_scratch);
        score_block(blocks[b], sum_scratch);
    }
}
