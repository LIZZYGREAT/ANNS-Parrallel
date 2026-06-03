#pragma once
#include "searcher.h"
#include "pq_distance.h"
#include <vector>
#include <limits>

struct SearchBuffers {
    std::vector<SearchWorkspace> workspaces;
    std::vector<std::vector<Candidate>> per_thread_topk;
    std::vector<float> thresholds;
    std::vector<Candidate> coarse_cands;
    std::vector<Candidate> merged;
    std::vector<uint32_t> rerank_ids;
    std::vector<float> rerank_dists;

    void prepare(int nt, int n_lists, int rerank_k) {
        const size_t cap = static_cast<size_t>(rerank_k) * 2;
        if (workspaces.size() < static_cast<size_t>(nt)) {
            workspaces.resize(static_cast<size_t>(nt));
        }
        if (per_thread_topk.size() < static_cast<size_t>(nt)) {
            per_thread_topk.resize(static_cast<size_t>(nt));
        }
        if (thresholds.size() < static_cast<size_t>(nt)) {
            thresholds.resize(static_cast<size_t>(nt));
        }
        const float inf = std::numeric_limits<float>::max();
        for (int t = 0; t < nt; ++t) {
            thresholds[static_cast<size_t>(t)] = inf;
            per_thread_topk[static_cast<size_t>(t)].clear();
            if (per_thread_topk[static_cast<size_t>(t)].capacity() < cap) {
                per_thread_topk[static_cast<size_t>(t)].reserve(cap);
            }
        }
        if (coarse_cands.size() < static_cast<size_t>(n_lists)) {
            coarse_cands.resize(static_cast<size_t>(n_lists));
        }
        const size_t merge_cap = cap * static_cast<size_t>(nt);
        if (merged.capacity() < merge_cap) {
            merged.reserve(merge_cap);
        }
        if (rerank_ids.size() < cap) {
            rerank_ids.resize(cap);
            rerank_dists.resize(cap);
        }
    }
};
