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

        std::vector<Candidate> coarse_cands(index->n_lists);
        if (predefined_probes != nullptr) {
            for (int pi = 0; pi < nprobe; ++pi) {
                coarse_cands[pi] = predefined_probes[pi];
            }
        } else {
            std::vector<float> coarse_dists(index->n_lists);
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
            
            for (int c = 0; c < index->n_lists; ++c) {
                coarse_cands[c] = {coarse_dists[c], static_cast<uint32_t>(c)};
            }
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(), cmp_asc);
        }


        std::vector<Candidate> all_cands;
        all_cands.reserve(static_cast<size_t>(nprobe) * top_k);

        float min_centroid_dist = coarse_cands[0].dist;
        
        for (int pi = 0; pi < nprobe; ++pi) {

            if (pi > 0 && coarse_cands[pi].dist > min_centroid_dist * 1.5f) {
                break; 
            }

            int list_id = static_cast<int>(coarse_cands[pi].id);
            auto* local_hnsw = index->hnsw_graphs[list_id];
            
            if (local_hnsw == nullptr) {
                continue;
            }

            auto local_res = local_hnsw->searchKnn(query, top_k);

            while (!local_res.empty()) {
                auto top_item = local_res.top();
                local_res.pop();
                
                Candidate c;
                c.dist = top_item.first;
                c.id = static_cast<uint32_t>(top_item.second);
                all_cands.push_back(c);
            }
        }

        if (all_cands.size() > static_cast<size_t>(top_k)) {
            std::nth_element(all_cands.begin(), all_cands.begin() + top_k, all_cands.end(), cmp_asc);
            all_cands.resize(top_k); 
        }

        std::priority_queue<Candidate> final_pq;
        for (const auto& c : all_cands) {
            final_pq.push(c);
        }

        return final_pq;
    }
};