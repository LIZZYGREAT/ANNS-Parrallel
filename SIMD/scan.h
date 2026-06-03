#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include "distance_neon.h"

// 专门用于全局缓冲的结构体
struct QueryResult {
    float dist;
    uint32_t id;
    
    // 重载小于号，用于 nth_element 寻找最小距离
    bool operator<(const QueryResult& other) const {
        return dist < other.dist;
    }
};

// 暴露给 main.cc 的新接口
inline std::priority_queue<std::pair<float, int>> flat_search_aosoa(
    const float* aosoa_base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    QueryResult* global_results
) {
    const size_t simd_width = 4;
    size_t num_blocks = base_number / simd_width;

    // 1. NEON 核心计算阶段
    for (size_t i = 0; i < num_blocks; ++i) {
        const float* current_base_block = aosoa_base + i * simd_width * vecdim;
        
        float32x4_t dists = compute_L2_distance_neon_aosoa(query, current_base_block, vecdim);
        
        // 使用 vgetq_lane_f32 从寄存器提取标量并存入全局缓冲
        global_results[i * 4 + 0].dist = vgetq_lane_f32(dists, 0);
        global_results[i * 4 + 0].id = i * 4 + 0;
        
        global_results[i * 4 + 1].dist = vgetq_lane_f32(dists, 1);
        global_results[i * 4 + 1].id = i * 4 + 1;
        
        global_results[i * 4 + 2].dist = vgetq_lane_f32(dists, 2);
        global_results[i * 4 + 2].id = i * 4 + 2;
        
        global_results[i * 4 + 3].dist = vgetq_lane_f32(dists, 3);
        global_results[i * 4 + 3].id = i * 4 + 3;
    }

    // 2. 尾部兜底逻辑（当 base_number 不是 4 的倍数时）
    size_t tail_start = num_blocks * simd_width;
    for (size_t i = tail_start; i < base_number; ++i) {
        float dist = 0.0f;
        const float* tail_base = aosoa_base + tail_start * vecdim + (i - tail_start) * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            float diff = query[d] - tail_base[d];
            dist += diff * diff;
        }
        global_results[i].dist = dist;
        global_results[i].id = i;
    }

    // 3. 延迟排序阶段：利用 O(N) 的 introselect 获取前 k 小的元素
    std::nth_element(global_results, global_results + k, global_results + base_number);

    // 4. 为了兼容框架原有的验证逻辑，将其重新封装入 max-heap 形式的 priority_queue 返回
    std::priority_queue<std::pair<float, int>> pq;
    for (size_t i = 0; i < k; ++i) {
        pq.push({global_results[i].dist, (int)global_results[i].id});
    }

    return pq;
}