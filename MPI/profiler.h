#pragma once
#include <chrono>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <pthread.h>

template<typename T = void>
class MicroProfilerT {
public:
    static std::map<std::string, double> times;
    static pthread_mutex_t mutex;

    struct Timer {
        std::string name;
        std::chrono::high_resolution_clock::time_point start;

        Timer(std::string n) : name(std::move(n)), start(std::chrono::high_resolution_clock::now()) {}

        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed = end - start;
            pthread_mutex_lock(&MicroProfilerT::mutex);
            MicroProfilerT::times[name] += elapsed.count();
            pthread_mutex_unlock(&MicroProfilerT::mutex);
        }
    };

    static void print_and_save(const std::string& filepath, size_t num_samples = 1) {
        std::ofstream fout(filepath);
        std::cerr << "========== Micro-Profiling Results ==========\n";
        if (num_samples > 1) {
            std::cerr << "Samples: " << num_samples << " (per-query avg shown)\n";
        }

        double total = 0.0;
        for (const auto& pair : times) {
            total += pair.second;
        }

        for (const auto& pair : times) {
            double pct = total > 0.0 ? (pair.second / total * 100.0) : 0.0;
            double avg = pair.second / static_cast<double>(num_samples);
            std::cerr << pair.first << ": " << pair.second << " us total, "
                      << avg << " us/avg (" << pct << "%)\n";
            if (fout.is_open()) {
                fout << pair.first << "," << pair.second << "," << avg << "\n";
            }
        }
        std::cerr << "Total profiled: " << total << " us";
        if (num_samples > 1) {
            std::cerr << " (" << total / num_samples << " us/query avg)";
        }
        std::cerr << "\n=============================================\n";
        if (fout.is_open()) fout.close();
    }

    static void reset() {
        times.clear();
    }
};

template<typename T>
std::map<std::string, double> MicroProfilerT<T>::times;

template<typename T>
pthread_mutex_t MicroProfilerT<T>::mutex = PTHREAD_MUTEX_INITIALIZER;

using MicroProfiler = MicroProfilerT<>;