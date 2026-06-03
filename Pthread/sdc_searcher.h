#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "pq_distance.h"
#include "fast_scan_kernel.h"
#include <algorithm>
#include <limits>
#include <vector>

class SDCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;
    const float* base_data;
    int rerank_ratio;
    
    AlignedBuffer<float> center_dist_table;

    template<typename Cmp>
    __attribute__((always_inline)) void score_block_slots(
        const FSBlock& block, const uint16_t* sum_arr,
        float base_dist, float scale, float& threshold,
        std::vector<Candidate>& topk, int rerank_k, Cmp cmp) const {
        #pragma GCC unroll 16
        for (int k = 0; k < 16; ++k) {
            uint32_t id = block.ids[k];
            if (id == 0xFFFFFFFF) continue;
            float approx_dist = base_dist + scale * sum_arr[k];
            if (approx_dist < threshold) {
                topk.push_back({approx_dist, id});
                if (topk.size() >= static_cast<size_t>(rerank_k * 2)) {
                    std::nth_element(topk.begin(), topk.begin() + rerank_k, topk.end(), cmp);
                    topk.resize(rerank_k);
                    threshold = std::max_element(topk.begin(), topk.begin() + rerank_k, cmp)->dist;
                }
            }
        }
    }

    template<typename Cmp>
    void scan_probe_list(int list_id, float coarse_dist, const float* query,
                         SearchWorkspace& ws, float& threshold,
                         std::vector<Candidate>& topk, int rerank_k, Cmp cmp) const {
        const InvertedList& cur_list = index->lists[list_id];
        if (cur_list.total_elements == 0) return;

        const float* cent = &index->ivf_centroids[list_id * index->d];
        residual_sub_d96(query, cent, ws.residual);

        // 1. 将当前的 Query 残差向量在线量化到最近的 PQ 质心编码
        for (int m = 0; m < FS_M; ++m) {
            const float* q_sub = ws.residual + m * FS_D_SUB;
            float min_dist = std::numeric_limits<float>::max();
            uint8_t best_k = 0;
            for (int k = 0; k < 16; ++k) {
                const float* c_sub = index->pq_centroids.data() + (m * 16 + k) * FS_D_SUB;
                float dist = compute_l2_sqr_d3(q_sub, c_sub);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = static_cast<uint8_t>(k);
                }
            }
            ws.query_code[m] = best_k;
        }

        // 2. 利用只读的 center_dist_table 快速组装当前倒排桶的 lut_f
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();

        for (int m = 0; m < FS_M; ++m) {
            uint8_t q_code = ws.query_code[m];
            const float* table_row = center_dist_table.data() + m * 256 + q_code * 16;
            #pragma GCC unroll 16
            for (int k = 0; k < 16; ++k) {
                float val = table_row[k];
                ws.lut_f[m * 16 + k] = val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }

        // 3. 归一化并进行 8 位整型定量
        float scale = (max_val - min_val) / 255.0f;
        float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        float base_dist = coarse_dist + FS_M * min_val;
        lut_float_to_u8(ws.lut_f, FS_M * 16, min_val, inv_scale, ws.lut_u8);

        const FSBlock* blocks = cur_list.blocks.data();
        const size_t nb = cur_list.blocks.size();
        auto score_fn = [&](const FSBlock& block, const uint16_t* sums) {
            score_block_slots(block, sums, base_dist, scale, threshold, topk, rerank_k, cmp);
        };
        fast_scan_list_batch(blocks, nb, ws.lut_u8, FS_M, ws.sum_arr, score_fn);
    }

public:
    SDCSearcher(const IVFPQIndex* idx, const float* base, int ratio = 20)
        : index(idx), base_data(base), rerank_ratio(ratio) {
        
        // 预计算所有子空间中，16个聚类中心两两之间的 L2 距离平方
        center_dist_table.resize(static_cast<size_t>(FS_M * 256));
        for (int m = 0; m < FS_M; ++m) {
            for (int i = 0; i < 16; ++i) {
                const float* c_i = index->pq_centroids.data() + (m * 16 + i) * FS_D_SUB;
                for (int j = 0; j < 16; ++j) {
                    const float* c_j = index->pq_centroids.data() + (m * 16 + j) * FS_D_SUB;
                    float dist = compute_l2_sqr_d3(c_i, c_j);
                    center_dist_table[static_cast<size_t>(m * 256 + i * 16 + j)] = dist;
                }
            }
        }
    }

    std::priority_queue<Candidate> search(const float* query, int top_k, int nprobe) override {
        int rerank_k = top_k * rerank_ratio;
        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        };

        // 通过 thread_local 为每个批处理线程圈出自己的绝对私有空间
        thread_local AlignedBuffer<float> local_coarse_dists;
        thread_local std::vector<Candidate> local_coarse_cands;
        thread_local SearchWorkspace local_ws;
        thread_local std::vector<Candidate> local_topk;
        thread_local std::vector<uint32_t> local_rerank_ids;
        thread_local std::vector<float> local_rerank_dists;

        if (local_coarse_dists.size() < static_cast<size_t>(index->n_lists)) {
            local_coarse_dists.resize(index->n_lists);
            local_coarse_cands.resize(index->n_lists);
        }
        if (local_topk.capacity() < static_cast<size_t>(rerank_k * 2)) {
            local_topk.reserve(rerank_k * 2);
            local_rerank_ids.resize(rerank_k * 2);
            local_rerank_dists.resize(rerank_k * 2);
        }
        local_topk.clear();

        // 1. 粗量化距离计算
        compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, local_coarse_dists.data());
        for (int c = 0; c < index->n_lists; ++c) {
            local_coarse_cands[static_cast<size_t>(c)] = {local_coarse_dists[c], static_cast<uint32_t>(c)};
        }

        // 2. 粗量化粗筛
        std::partial_sort(local_coarse_cands.begin(), local_coarse_cands.begin() + nprobe,
                          local_coarse_cands.end(), cmp_asc);

        // 3. 倒排桶探查
        float threshold = std::numeric_limits<float>::max();
        for (int pi = 0; pi < nprobe; ++pi) {
            int list_id = static_cast<int>(local_coarse_cands[pi].id);
            scan_probe_list(list_id, local_coarse_cands[pi].dist, query, local_ws,
                            threshold, local_topk, rerank_k, cmp_asc);
        }

        // 4. 局部 Top-K 截断
        if (local_topk.size() > static_cast<size_t>(rerank_k)) {
            std::nth_element(local_topk.begin(), local_topk.begin() + rerank_k, local_topk.end(), cmp_asc);
            local_topk.resize(rerank_k);
        }

        // 5. 精确重排
        const int n = static_cast<int>(local_topk.size());
        for (int i = 0; i < n; ++i) local_rerank_ids[static_cast<size_t>(i)] = local_topk[static_cast<size_t>(i)].id;
        
        rerank_batch_d96(query, base_data, index->d, local_rerank_ids.data(), local_rerank_dists.data(), n);
        
        for (int i = 0; i < n; ++i) local_topk[static_cast<size_t>(i)].dist = local_rerank_dists[static_cast<size_t>(i)];
        
        if (local_topk.size() > static_cast<size_t>(top_k)) {
            std::nth_element(local_topk.begin(), local_topk.begin() + top_k, local_topk.end(), cmp_asc);
            local_topk.resize(top_k);
        }

        // 6. 构筑最终优先队列返回
        std::priority_queue<Candidate> final_pq;
        for (const auto& cand : local_topk) {
            final_pq.push(cand);
        }
        return final_pq;
    }
};