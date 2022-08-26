cmake_minimum_required(VERSION 3.20)
project(learn_cuda_jit CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)

find_package(fmt)

add_executable(learn_cuda_jit main.cu UDF)
set_target_properties(learn_cuda_jit PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(learn_cuda_jit
        PUBLIC cuda
        PUBLIC cudart
        PUBLIC nvrtc
        PUBLIC fmt::fmt)