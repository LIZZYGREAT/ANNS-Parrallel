#pragma once

#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include "simd_l2.h"
#include "thread_pool.h"

class KMeans {
public:
    int d;
    int k;
    std::vector<float> centroids;

    KMeans(int dim, int num_clusters) : d(dim), k(num_clusters) {
        centroids.resize(k * d, 0.0f);
    }

    void train(const float* train_data, size_t n, int max_iter = 20) {
        if (n == 0 || k == 0) return;

        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int c = 0; c < k; ++c) {
            size_t idx = indices[c];
            std::memcpy(&centroids[c * d], &train_data[idx * d], d * sizeof(float));
        }

        int max_threads = tp::get_num_threads();
        std::vector<float> local_new_centroids(static_cast<size_t>(max_threads) * k * d, 0.0f);
        std::vector<int> local_counts(static_cast<size_t>(max_threads) * k, 0);

        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(local_new_centroids.begin(), local_new_centroids.end(), 0.0f);
            std::fill(local_counts.begin(), local_counts.end(), 0);

            tp::parallel_region([&](int tid) {
                float* my_centroids_sum = &local_new_centroids[static_cast<size_t>(tid) * k * d];
                int* my_counts = &local_counts[static_cast<size_t>(tid) * k];
                const size_t chunk = (n + static_cast<size_t>(max_threads) - 1) / static_cast<size_t>(max_threads);
                const size_t i0 = static_cast<size_t>(tid) * chunk;
                const size_t i1 = std::min(n, i0 + chunk);
                
                std::vector<float> dist_buf(k);

                for (size_t i = i0; i < i1; ++i) {
                    const float* current_data = &train_data[i * d];
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;
                    if (d == 96) {
                        compute_all_L2_sqr_d96(current_data, centroids.data(), k, dist_buf.data());
                        for (int c = 0; c < k; ++c) {
                            if (dist_buf[c] < min_dist) {
                                min_dist = dist_buf[c];
                                best_c = c;
                            }
                        }
                    } else {
                        for (int c = 0; c < k; ++c) {
                            float dist = compute_L2_sqr(current_data, &centroids[c * d], d);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_c = c;
                            }
                        }
                    }
                    
                    my_counts[best_c]++;
                    for (int j = 0; j < d; ++j) {
                        my_centroids_sum[best_c * d + j] += current_data[j];
                    }
                }
            });

            tp::parallel_for_static(0, static_cast<size_t>(k), [&](size_t c) {
                int total_count = 0;
                std::vector<float> global_sum(d, 0.0f);
                for (int t = 0; t < max_threads; ++t) {
                    total_count += local_counts[static_cast<size_t>(t) * k + static_cast<int>(c)];
                    for (int j = 0; j < d; ++j) {
                        global_sum[j] += local_new_centroids[static_cast<size_t>(t) * k * d + c * d + j];
                    }
                }
                if (total_count == 0) {
                    size_t rand_idx = static_cast<size_t>(std::rand()) % n;
                    std::memcpy(&centroids[c * d], &train_data[rand_idx * d], d * sizeof(float));
                } else {
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = global_sum[j] / total_count;
                    }
                }
            });
        }
    }
};