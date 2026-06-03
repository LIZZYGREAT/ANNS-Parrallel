#pragma once
#include "searcher.h"
#include "hnsw_index.h"
#include <queue>
#include <vector>

class HNSWSearcher {
private:
    const HNSWIndex* index;

public:
    HNSWSearcher(const HNSWIndex* idx) : index(idx) {}

    const HNSWIndex* get_index() const { return index; }

    std::priority_queue<Candidate> search(
        const float* query, 
        int top_k, 
        int ef_search = 50) {
        
        int actual_ef = std::max(top_k, ef_search);
        index->hnsw_graph->setEf(actual_ef);

        auto local_res = index->hnsw_graph->searchKnn(query, top_k);

        std::priority_queue<Candidate> final_pq;
        
        while (!local_res.empty()) {
            auto top_item = local_res.top();
            local_res.pop();
            
            Candidate c;
            c.dist = top_item.first;
            c.id = static_cast<uint32_t>(top_item.second); 
            final_pq.push(c);
        }

        return final_pq;
    }
};