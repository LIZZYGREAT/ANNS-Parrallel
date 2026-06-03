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

class IVFPQIndex;

class BaseSearcher {
public:
    virtual ~BaseSearcher() = default;
    
    virtual std::priority_queue<Candidate> search(
        const float* query, 
        int top_k, 
        int nprobe,
        const Candidate* predefined_probes = nullptr) = 0; 
    virtual const IVFPQIndex* get_index() const = 0;
};