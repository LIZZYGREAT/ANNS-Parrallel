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