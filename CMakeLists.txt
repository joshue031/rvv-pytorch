cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(rvv_pytorch)

find_package(Torch REQUIRED)

# Define our library target
add_library(rvv_pytorch SHARED rvv.cpp)

# Enable C++14
target_compile_features(rvv_pytorch PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(rvv_pytorch "${TORCH_LIBRARIES}")

set(CMAKE_CXX_COMPILER "/usr/bin/clang-15")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")
