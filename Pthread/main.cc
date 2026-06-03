#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "thread_pool.h"
#include <stdio.h>
#include <queue>
#include <ctime>

#include "profiler.h"
#include "ivfpq_index.h"
#include "adc_searcher.h"
#include "sdc_searcher.h"

using namespace std;

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
       << "_IVFPQ_ADC-SDC_nlist" << n_lists << "_M" << FS_M << "_rerank20";
    std::string dir = ss.str();
    mkdir("runs", 0755);
    mkdir(dir.c_str(), 0755);
    return dir;
}

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[Error] Cannot open file " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult {
    float recall;
    int64_t latency;
};

void run_evaluation(int thread_count, int nprobe, BaseSearcher* searcher,
                    const float* test_query, const int* test_gt,
                    size_t test_number, size_t vecdim, size_t test_gt_d, size_t k,
                    std::ofstream& csv_file, const std::string& method_name,
                    const std::string& out_dir) {

    tp::set_num_threads(thread_count);
    std::vector<SearchResult> results(test_number);

    // 记录整个 Batch 批处理的起止时间，用于计算真正的 QPS
    struct timeval batch_val;
    gettimeofday(&batch_val, NULL);

    tp::parallel_region([&](int tid) {
        // 静态分块：让各个线程领走属于自己的一批 Query
        const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
        const size_t start_idx = static_cast<size_t>(tid) * chunk;
        const size_t end_idx = std::min(test_number, start_idx + chunk);

        for (size_t i = start_idx; i < end_idx; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);

            auto res = searcher->search(test_query + i * vecdim, k, nprobe);

            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(size_t j = 0; j < k; ++j){
                gtset.insert(test_gt[j + i*test_gt_d]);
            }

            size_t acc = 0;
            while (!res.empty()) {
                if(gtset.count(res.top().id)) ++acc;
                res.pop();
            }

            results[i] = {(float)acc / k, diff};
        }
    });

    struct timeval new_batch_val;
    gettimeofday(&new_batch_val, NULL);
    const unsigned long Converter = 1000 * 1000;
    int64_t batch_diff_us = (new_batch_val.tv_sec * Converter + new_batch_val.tv_usec) - (batch_val.tv_sec * Converter + batch_val.tv_usec);
    
    // 计算宏观指标
    float final_recall = 0, final_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {
        final_recall += results[i].recall;
        final_latency += results[i].latency;
    }
    final_recall /= test_number;
    final_latency /= test_number;
    
    float qps = (float)test_number / ((float)batch_diff_us / 1000000.0f);

    std::cerr << "[Result] Method: " << method_name << " | Threads: " << thread_count
              << " | NProbe: " << nprobe << " | Recall: " << final_recall
              << " | Avg Latency: " << final_latency << " us"
              << " | QPS: " << qps << "\n";

    csv_file << method_name << "," << thread_count << "," << nprobe << ","
             << final_recall << "," << final_latency << "," << qps << "\n";
}

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 200; 
    
    const size_t k = 10;
    const int n_lists = 1024;
    const int rerank_ratio = 30;

    std::string out_dir = make_run_dir(n_lists);
    std::cerr << "[System] Output dir: " << out_dir << "\n";

    IVFPQIndex index(vecdim, n_lists);

    std::cerr << "[System] Building Index...\n";
    index.build(base, base_number);

    ADCSearcher adc_searcher(&index, base, rerank_ratio);
    SDCSearcher sdc_searcher(&index, base, rerank_ratio);

    std::ofstream csv_file(out_dir + "/ivfpq_tradeoff.csv");

    csv_file << "Method,Threads,NProbe,Recall@10,Latency(us),QPS\n";

    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<int> nprobe_configs = {8, 16, 32, 64, 128};

    std::cerr << "\n[System] Starting Automated Grid Search Evaluation (Batch Parallelism)...\n";

    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            std::cerr << "\n>>> Running ADC config: Threads=" << t << ", NProbe=" << probe << "\n";
            run_evaluation(t, probe, &adc_searcher, test_query, test_gt,
                           test_number, vecdim, test_gt_d, k, csv_file, "ADC", out_dir);
        }
    }

    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            std::cerr << "\n>>> Running SDC config: Threads=" << t << ", NProbe=" << probe << "\n";
            run_evaluation(t, probe, &sdc_searcher, test_query, test_gt,
                           test_number, vecdim, test_gt_d, k, csv_file, "SDC", out_dir);
        }
    }

    csv_file.close();

    {
        std::ofstream meta(out_dir + "/run_info.txt");
        meta << "algorithm=IVFPQ_ADC-SDC\n";
        meta << "n_lists=" << n_lists << "\n";
        meta << "M=" << FS_M << "\n";
        meta << "rerank_ratio=" << rerank_ratio << "\n";
        meta << "test_queries=" << test_number << "\n";
        meta << "k=" << k << "\n";
    }

    std::cerr << "\n[System] All evaluations completed. Results in " << out_dir << "\n";

    tp::shutdown_pool();

    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    return 0;
}