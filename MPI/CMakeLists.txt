cmake_minimum_required(VERSION 3.10)
project(Parallel_ANN_MPI)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 配置高性能编译优化参数
# (-O3 开启深度优化, -march=native 自动启用本地 CPU 的 AVX2/NEON/SIMD 指令集)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# 寻找 MPI 库支持
find_package(MPI REQUIRED)
# 寻找 Threads 库 (Pthreads)
find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 添加可执行文件
add_executable(main_mpi main.cc)

# 链接 MPI 和 Pthread
target_link_libraries(main_mpi PRIVATE MPI::MPI_CXX Threads::Threads)