#pragma once
#include <queue>
#include <vector>
#include <cstdint>

struct Candidate {
    float dist;
    uint32_t id;
    bool operator<(const Candidate& other) const {
        return dist > other.dist;
    }
};

class BaseSearcher {
public:
    virtual ~BaseSearcher() = default;
    
    virtual std::priority_queue<Candidate> search(
        const float* query, 
        int top_k, 
        int nprobe) = 0;
};