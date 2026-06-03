#pragma once
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

struct SQQuantizer {
    float v_min;
    float v_max;
    float scale;
    float inv_scale;

    SQQuantizer() : v_min(0), v_max(0), scale(0), inv_scale(0) {}

    void train(const float* base_data, size_t num_vectors, size_t dim) {
        size_t total_elements = num_vectors * dim;
        if (total_elements == 0) return;

        double sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i) {
            sum += base_data[i];
        }
        float mean = static_cast<float>(sum / total_elements);

        double variance_sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i) {
            float diff = base_data[i] - mean;
            variance_sum += diff * diff;
        }
        float std_dev = std::sqrt(static_cast<float>(variance_sum / total_elements));

        float k = 3.0f; 
        v_min = mean - k * std_dev;
        v_max = mean + k * std_dev;

        if (v_max - v_min < 1e-6f) {
            v_max = v_min + 1e-6f;
        }
        
        scale = 255.0f / (v_max - v_min);
        inv_scale = (v_max - v_min) / 255.0f;
        
        std::cerr << "[SQ Info] Statistical Train Complete.\n"
                  << "  Mean: " << mean << ", StdDev: " << std_dev << "\n"
                  << "  Clipping Range: [" << v_min << ", " << v_max << "]\n"
                  << "  Scale: " << scale << "\n";
    }

    inline void encode(const float* input, uint8_t* output, size_t dim) const {
        for (size_t i = 0; i < dim; ++i) {
            float val = (input[i] - v_min) * scale;
            
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            
            output[i] = static_cast<uint8_t>(val + 0.5f); 
        }
    }

    void encode_batch(const float* input_data, uint8_t* output_data, size_t num_vectors, size_t dim) const {
        for (size_t i = 0; i < num_vectors; ++i) {
            encode(input_data + i * dim, output_data + i * dim, dim);
        }
    }
};

inline uint32_t compute_L2_distance_sq_neon(
    const uint8_t* query_ptr,
    const uint8_t* base_ptr,
    const size_t dim
) __attribute__((always_inline));

inline uint32_t compute_L2_distance_sq_neon(
    const uint8_t* query_ptr,
    const uint8_t* base_ptr,
    const size_t dim
) {

    uint32x4_t sum_u32 = vdupq_n_u32(0);

    for (size_t d = 0; d < dim; d += 16) {
        uint8x16_t q = vld1q_u8(query_ptr + d);
        uint8x16_t b = vld1q_u8(base_ptr + d);

        uint8x16_t diff = vabdq_u8(q, b);

        uint8x8_t diff_lo = vget_low_u8(diff);
        uint8x8_t diff_hi = vget_high_u8(diff);

        uint16x8_t sq_lo = vmull_u8(diff_lo, diff_lo);
        uint16x8_t sq_hi = vmull_u8(diff_hi, diff_hi);


        sum_u32 = vpadalq_u16(sum_u32, sq_lo);
        sum_u32 = vpadalq_u16(sum_u32, sq_hi);
    }

    uint32x2_t sum_half = vadd_u32(vget_low_u32(sum_u32), vget_high_u32(sum_u32));
    uint32_t final_distance = vget_lane_u32(sum_half, 0) + vget_lane_u32(sum_half, 1);
    
    return final_distance;
}