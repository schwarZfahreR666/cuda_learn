cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(cuda_learn CXX C CUDA)
# compute表示编译出的ptx版本，还需要jit才能运行
# sm是实际二进制的版本
set(CMAKE_CUDA_FLAGS "-arch=compute_89 -code=sm_89 -g -G -O3")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BUILD_DIR})

include_directories(./include)

add_subdirectory(0_hello_world)
add_subdirectory(1_sum_arrays)
add_subdirectory(2_sum_matrix)
add_subdirectory(3_recursiveReduce)
add_subdirectory(4_sgemm)