#pragma once
#include "searcher.h"
#include "ivf_hnsw_index.h"
#include <queue>
#include <vector>
#include <algorithm>

class IVFHNSWSearcher {
private:
    const IVFHNSWIndex* index;

public:
    IVFHNSWSearcher(const IVFHNSWIndex* idx) : index(idx) {}

    const IVFHNSWIndex* get_index() const { return index; }

    std::priority_queue<Candidate> search(
        const float* query, 
        int top_k, 
        int nprobe, 
        const Candidate* predefined_probes = nullptr) {
        
        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist; 
        };

        // 用于维护最终 top_k 结果的大顶堆
        std::priority_queue<Candidate> final_pq;

        // 1. 粗检索阶段：寻找距离 Query 最近的 nprobe 个 IVF 桶
        std::vector<Candidate> coarse_cands(index->n_lists);
        if (predefined_probes != nullptr) {
            // 如果由外部 MPI 的 Rank 0 统一下发了探查目标，直接使用
            for (int pi = 0; pi < nprobe; ++pi) {
                coarse_cands[pi] = predefined_probes[pi];
            }
        } else {
            // 否则本地计算 Query 到所有 centroids 的距离
            std::vector<float> coarse_dists(index->n_lists);
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
            
            for (int c = 0; c < index->n_lists; ++c) {
                coarse_cands[c] = {coarse_dists[c], static_cast<uint32_t>(c)};
            }
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(), cmp_asc);
        }

        // 2. 细检索阶段：在对应的 HNSW 图中进行搜寻
        for (int pi = 0; pi < nprobe; ++pi) {
            int list_id = static_cast<int>(coarse_cands[pi].id);
            
            auto* local_hnsw = index->hnsw_graphs[list_id];
            
            if (local_hnsw == nullptr) {
                continue;
            }

            auto local_res = local_hnsw->searchKnn(query, top_k);

            // 3. 归并局部结果池到最终的全局堆中
            while (!local_res.empty()) {
                auto top_item = local_res.top();
                local_res.pop();
                
                // 将结果映射转换放入 final_pq
                Candidate c;
                c.dist = top_item.first;
                c.id = top_item.second;
                
                final_pq.push(c);
                if (final_pq.size() > static_cast<size_t>(top_k)) {
                    final_pq.pop();
                }
            }
        }
        return final_pq;
    }
};