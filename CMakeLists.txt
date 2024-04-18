cmake_minimum_required(VERSION 3.10)
project(TP_GPU C CUDA)
set(CMAKE_C_STANDARD 11)

set(COMMON_INCLUDES defs.h)
include_directories(.)
add_executable(checker
        checker.c
        ${COMMON_INCLUDES}
)
add_executable(sequential
        sequential.c
        ${COMMON_INCLUDES}
)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    target_link_libraries(checker PUBLIC ${MATH_LIBRARY})
    target_link_libraries(sequential PUBLIC ${MATH_LIBRARY})
endif ()

target_compile_options(checker PRIVATE -O3 -march=native -mtune=native)
target_compile_options(sequential PRIVATE -O3 -march=native -mtune=native)

# CUDA
if ($ENV{CLION_IDE})
    include_directories("/opt/cuda/include")
endif ()
find_package(CUDAToolkit REQUIRED)
if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()
add_executable(simple
        simple.cu
        CUDA_common.h
        ${COMMON_INCLUDES}
)
target_compile_options(simple PRIVATE "--generate-code arch=compute_60,code=sm_60 -O3")
