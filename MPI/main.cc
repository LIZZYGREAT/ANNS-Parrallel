#include <mpi.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <ctime>
#include "profiler.h"
#include "ivfpq_index.h"
#include "adc_searcher.h"
#include "sdc_searcher.h"
#include "thread_pool.h"

using namespace std;

struct DistancePair {
    float dist;
    int id;
};

std::string make_run_dir(int n_lists) {
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::localtime(&now);
    std::stringstream ss;
    ss << "files/"
       << (1900 + ltm->tm_year)
       << std::setw(2) << std::setfill('0') << (1 + ltm->tm_mon)
       << std::setw(2) << std::setfill('0') << ltm->tm_mday << "_"
       << std::setw(2) << std::setfill('0') << ltm->tm_hour
       << std::setw(2) << std::setfill('0') << ltm->tm_min
       << std::setw(2) << std::setfill('0') << ltm->tm_sec
       << "_IVFPQ_ADC-SDC_nlist" << n_lists;
    return ss.str();
}

template<typename T>
T* LoadData(const std::string& filename, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    in.read((char*)&num, sizeof(int));
    in.read((char*)&dim, sizeof(int));
    
    T* data = new T[num * dim];
    in.read((char*)data, num * dim * sizeof(T));
    in.close();
    return data;
}

template<typename SearcherType>
void run_mpi_evaluation(int thread_count, int nprobe, SearcherType* searcher,
                        const float* test_query, const int* test_gt,
                        size_t test_number, size_t vecdim, size_t test_gt_d, size_t k,
                        std::ofstream& csv_file, const std::string& method_name,
                        const std::string& out_dir, int local_base_number,
                        int rank, int size) {

    tp::set_num_threads(thread_count);
    std::vector<DistancePair> local_results(test_number * k);

    std::vector<Candidate> all_probes(test_number * nprobe);
    if (rank == 0) {
        tp::parallel_region([&](int tid) {
            const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
            const size_t start_idx = static_cast<size_t>(tid) * chunk;
            const size_t end_idx = std::min(test_number, start_idx + chunk);
            
            const IVFPQIndex* idx = searcher->get_index();
            int n_lists = idx->n_lists;
            std::vector<float> dists(n_lists);
            std::vector<Candidate> coarse_cands(n_lists);

            for (size_t i = start_idx; i < end_idx; ++i) {
                const float* q = test_query + i * vecdim;
                compute_all_L2_sqr_d96(q, idx->ivf_centroids.data(), n_lists, dists.data());
                for (int c = 0; c < n_lists; ++c) coarse_cands[c] = {dists[c], static_cast<uint32_t>(c)};
                std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(), 
                                  [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                for (int p = 0; p < nprobe; ++p) {
                    all_probes[i * nprobe + p] = coarse_cands[p];
                }
            }
        });
    }

    // Rank 0 广播所有 Query 的倒排桶探查目标
    MPI_Bcast(all_probes.data(), test_number * nprobe * sizeof(Candidate), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    struct timeval start_local, end_local;
    gettimeofday(&start_local, NULL);

    tp::parallel_region([&](int tid) {
        const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
        const size_t start_idx = static_cast<size_t>(tid) * chunk;
        const size_t end_idx = std::min(test_number, start_idx + chunk);

        for (size_t i = start_idx; i < end_idx; ++i) {
            const float* query_vec = test_query + i * vecdim;
            // 获取 Rank 0 下发的命中倒排桶 ID
            const Candidate* my_probes = &all_probes[i * nprobe]; 
            
            std::priority_queue<Candidate> pq_res = searcher->search(query_vec, k, nprobe, my_probes);

            int idx = k - 1;
            while (!pq_res.empty() && idx >= 0) {
                Candidate c = pq_res.top();
                pq_res.pop();
                DistancePair dp;
                dp.dist = c.dist;
                dp.id = rank * local_base_number + c.id; 
                local_results[i * k + idx] = dp;
                idx--;
            }
        }
    });

    gettimeofday(&end_local, NULL);
    double local_time_us = (end_local.tv_sec - start_local.tv_sec) * 1000000.0 + (end_local.tv_usec - start_local.tv_usec);
    double max_local_time_us = 0.0;

    MPI_Reduce(&local_time_us, &max_local_time_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    struct timeval start_gather, end_gather;
    gettimeofday(&start_gather, NULL);

    std::vector<DistancePair> global_gathered_results;
    if (rank == 0) {
        global_gathered_results.resize(test_number * k * size);
    }

    MPI_Gather(local_results.data(), test_number * k * sizeof(DistancePair), MPI_BYTE,
               global_gathered_results.data(), test_number * k * sizeof(DistancePair), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> final_global_indices(test_number * k);
        for (size_t q = 0; q < test_number; ++q) {
            std::vector<DistancePair> candidates;
            candidates.reserve(size * k);
            for (int r = 0; r < size; ++r) {
                int offset = r * (test_number * k) + q * k;
                for (size_t j = 0; j < k; ++j) {
                    candidates.push_back(global_gathered_results[offset + j]);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const DistancePair& a, const DistancePair& b) {
                return a.dist < b.dist;
            });
            for (size_t j = 0; j < k; ++j) {
                final_global_indices[q * k + j] = candidates[j].id;
            }
        }

        gettimeofday(&end_gather, NULL);
        double gather_merge_time_us = (end_gather.tv_sec - start_gather.tv_sec) * 1000000.0 + (end_gather.tv_usec - start_gather.tv_usec);
        
        double total_recall = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) {
                gtset.insert(test_gt[q * test_gt_d + j]);
            }
            size_t match_count = 0;
            for (size_t j = 0; j < k; ++j) {
                if (gtset.count(final_global_indices[q * k + j])) match_count++;
            }
            total_recall += (double)match_count / k;
        }

        float final_recall = total_recall / test_number;
        float local_latency_per_query = max_local_time_us / test_number;
        float qps = test_number / (max_local_time_us / 1000000.0f); 

        std::cerr << fixed << setprecision(4);
        std::cerr << "[Result] Method: " << method_name << " | Threads: " << thread_count
                  << " | NProbe: " << nprobe << " | Recall: " << final_recall
                  << " | Local Latency: " << local_latency_per_query << " us"
                  << " | QPS: " << qps 
                  << " | (MPI Overhead: " << (gather_merge_time_us / test_number) << " us/q)\n";


        csv_file << method_name << "," 
                 << thread_count << "," 
                 << nprobe << ","
                 << final_recall << "," 
                 << local_latency_per_query << "," 
                 << qps << "," 
                 << (gather_merge_time_us / test_number) << "\n";
        csv_file.flush(); 
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string data_path = "./anndata/";

    int n_lists = 1024; 
    int k = 10;         

    int base_number = 0, vecdim = 0;
    int test_number = 0, query_dim = 0;
    int test_gt_d = 0;

    float* full_base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;

    std::string out_dir = "";

    if (rank == 0) {
        std::cerr << "[System] Rank 0 loading high-dimensional data sets...\n";
        full_base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, query_dim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);

        if (!full_base || !test_query || !test_gt) {
            std::cerr << "CRITICAL: Data files missing or corrupted!\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        test_number = 200;
        
        out_dir = make_run_dir(n_lists);
        mkdir("files", 0777);
        mkdir(out_dir.c_str(), 0777);
        std::cerr << "[System] Output telemetry files will be saved to: " << out_dir << "\n";
    }

    MPI_Bcast(&base_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_base_number = base_number / size;
    float* local_base = new float[local_base_number * vecdim];

    MPI_Scatter(full_base, local_base_number * vecdim, MPI_FLOAT,
                local_base, local_base_number * vecdim, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank != 0) {
        test_query = new float[test_number * vecdim];
    }
    MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0 && full_base != nullptr) {
        delete[] full_base;
        full_base = nullptr;
    }

    IVFPQIndex index(vecdim, n_lists);
    index.build(local_base, local_base_number, rank, size);

    int rerank_ratio = 100; 
    ADCSearcher adc_searcher(&index, local_base, rerank_ratio);
    SDCSearcher sdc_searcher(&index, local_base, rerank_ratio);

    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open(out_dir + "/results.csv");
        csv_file << "Algorithm,Threads,NProbe,Recall,LocalLatency_us,QPS,MPIOverhead_us\n";
    }

    // 网格搜索评测配置集
    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<int> nprobe_configs = {8, 16, 32, 64, 128};

    if (rank == 0) {
        std::cerr << "\n[System] Starting Automated Grid Search Evaluation (MPI Distributed Cluster Parallelism)...\n";
    }

    // 评测 ADC 算法
    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            run_mpi_evaluation(t, probe, &adc_searcher, test_query, test_gt,
                               test_number, vecdim, test_gt_d, k, csv_file, 
                               "ADC", out_dir, local_base_number, rank, size);
        }
    }

    // 评测 SDC 算法
    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            run_mpi_evaluation(t, probe, &sdc_searcher, test_query, test_gt,
                               test_number, vecdim, test_gt_d, k, csv_file, 
                               "SDC", out_dir, local_base_number, rank, size);
        }
    }

    if (rank == 0) {
        csv_file.close();
        std::ofstream meta(out_dir + "/run_info.txt");
        meta << "algorithm=IVFPQ_ADC-SDC\n";
        meta << "n_lists=" << n_lists << "\n";
        meta << "mpi_size=" << size << "\n";
        meta << "local_base_per_node=" << local_base_number << "\n";
        meta.close();
        std::cerr << "\n[System] Evaluation pipeline completed successfully. All data stored safely.\n";
        
        delete[] test_gt;
    }

    delete[] local_base;
    delete[] test_query;

    MPI_Finalize();
    return 0;
}