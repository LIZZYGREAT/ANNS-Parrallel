#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "pq_distance.h"
#include "fast_scan_kernel.h"
#include <algorithm>
#include <limits>
#include <vector>

class ADCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;
    const float* base_data;
    int rerank_ratio;

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
                    threshold = std::max_element(topk.begin(), topk.end(), cmp)->dist;
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

        float min_val, max_val;
        pq_build_adc_lut(ws.residual, FS_M, index->d_sub,
                         index->pq_centroids.data(), ws.lut_f,
                         min_val, max_val);
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
    ADCSearcher(const IVFPQIndex* idx, const float* base, int ratio = 20)
        : index(idx), base_data(base), rerank_ratio(ratio) {
    }

    std::priority_queue<Candidate> search(const float* query, int top_k, int nprobe) override {
        int rerank_k = top_k * rerank_ratio;
        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        };

        //使用 thread_local 保证每个并发处理 Query 的线程都有自己专属的内存空间，绝不打架
        thread_local AlignedBuffer<float> local_coarse_dists;
        thread_local std::vector<Candidate> local_coarse_cands;
        thread_local SearchWorkspace local_ws;
        thread_local std::vector<Candidate> local_topk;
        thread_local std::vector<uint32_t> local_rerank_ids;
        thread_local std::vector<float> local_rerank_dists;

        // 仅在线程第一次接客时分配内存
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

        // 2. 粗量化排序
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

        // 6. 构造最终结果
        std::priority_queue<Candidate> final_pq;
        for (const auto& cand : local_topk) {
            final_pq.push(cand);
        }
        return final_pq;
    }
};