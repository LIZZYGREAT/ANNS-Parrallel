#pragma once
#include "searcher.h"
#include "hnsw_hnsw_index.h"
#include <queue>
#include <vector>
#include <algorithm>

class HNSWHNSWSearcher {
private:
    const HNSWHNSWIndex* index;

public:
    HNSWHNSWSearcher(const HNSWHNSWIndex* idx) : index(idx) {}

    const HNSWHNSWIndex* get_index() const { return index; }

    std::priority_queue<Candidate> search(
        const float* query, 
        int top_k, 
        int nprobe) const { 
        
        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist; 
        };

        auto top_res = index->top_hnsw->searchKnn(query, nprobe);
        
        thread_local std::vector<int> target_buckets;
        target_buckets.clear();
        if (target_buckets.capacity() < static_cast<size_t>(nprobe)) {
            target_buckets.reserve(nprobe);
        }
        
        while (!top_res.empty()) {
            target_buckets.push_back(static_cast<int>(top_res.top().second));
            top_res.pop();
        }

        thread_local std::vector<Candidate> all_cands;
        all_cands.clear();
        size_t required_capacity = static_cast<size_t>(nprobe) * top_k;
        if (all_cands.capacity() < required_capacity) {
            all_cands.reserve(required_capacity);
        }

        for (int bucket_id : target_buckets) {
            auto* local_hnsw = index->bottom_hnsws[bucket_id];
            
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