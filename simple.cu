/**
 * HPC - M2 Data Science - Univ. Lille
 * Authors: C. Bouillaguet and P. Fortin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef ELEMENTS_PER_BLOCK
#define ELEMENTS_PER_BLOCK 256
#endif

// #define PROFILING
#include "profile_section.h"
#include "CUDA_common.h"
#include "defs.h"

__global__ void vecMatMultKernel(
        REAL_T *A,
        REAL_T *X,
        REAL_T *Y,
        int n
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int j;
        REAL_T tmp;

        // Split line addition (atomic) -> Occupancy
        tmp = 0;
        for (j = 0; j < n; j++) {
            tmp += A[i * n + j] * X[j];
        }
        Y[i] = tmp;
    }
}

__global__ void normSumKernel(
        REAL_T *Y,
        int n,
        REAL_T *odata
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        odata[i] = Y[i] * Y[i];
    } else {
        odata[i] = 0;
    }
}

__global__ void normalizeYKernel(
        REAL_T *Y,
        REAL_T *norm,
        int n
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        REAL_T inv_norm = 1.0 / sqrt(*norm);
        Y[i] *= inv_norm;
    }
}

__global__ void errorKernel(
        REAL_T *Y,
        REAL_T *X,
        int n,
        REAL_T *odata
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double delta = X[i] - Y[i];
        odata[i] = delta * delta;
    } else {
        odata[i] = 0;
    }
}

__global__ void reduce1(
        REAL_T *indata,
        REAL_T *odata,
        int size
) {
    __shared__ float block[ELEMENTS_PER_BLOCK];
    unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int i = threadIdx.x;
    if (globalIndex < size)
        block[i] = indata[globalIndex];
    else
        block[i] = 0;

    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
    {
        if (i < j)
            block[i] += block[i + j];

        __syncthreads();
    }
    if (i == 0)
        odata[blockIdx.x] = block[0];
}

void reduceArray(REAL_T *&d_input_data, REAL_T *&d_output_data, int size) {
    // Reduce
    int num_input = size;
    int num_output;
    const int threadsPerBlock = ELEMENTS_PER_BLOCK;
    do {
        num_output = num_input / threadsPerBlock;
        if (num_input % threadsPerBlock)
            num_output++;
        reduce1<<<num_output, threadsPerBlock>>>(d_input_data, d_output_data, num_input);
        num_input = num_output;
        // swap in and out and reduce
        if (num_output > 1) {
            reduce1<<<num_output, threadsPerBlock>>>(d_output_data, d_input_data, num_input);
        }
    } while(num_output > 1);
}


int main(int argc, char **argv) {
    int i, n;
    long long size;
    REAL_T error, norm;
    REAL_T *A, *X, *Y;
    double start_time, total_time;
    int n_iterations;
    FILE *output;

    if (argc < 2) {
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoi(argv[1]);
    size = (long long) n * n * sizeof(REAL_T);
    printf("Matrix size: %.3f G\n", (double) size / 1073741824.);

    /*** Matrix and vector allocation ***/
    A = (REAL_T *) malloc(size);
    if (A == NULL) {
        perror("Unable to allocate the matrix");
        exit(1);
    }
    X = (REAL_T *) malloc(n * sizeof(REAL_T));
    Y = (REAL_T *) malloc(n * sizeof(REAL_T));
    if ((X == NULL) || (Y == NULL)) {
        perror("Unable to allocate the vectors");
        exit(1);
    }
    /*** Initializing the matrix and x ***/
    // TODO: Kernel pour initialisation des matrices
    for (i = 0; i < n; i++) {
        init_row(A, i, n);
    }

    for (i = 0; i < n; i++) {
        X[i] = 1.0 / n;
    }

    // Kernel dimensions
    dim3 gridSize(ceil((double) n / ELEMENTS_PER_BLOCK));
    dim3 blockSize(ELEMENTS_PER_BLOCK);

    // alloc and transfer data to gpu
    printf("Allocating device memory... ");
    REAL_T *d_A, *d_X, *d_Y;
    REAL_T *d_error, *d_norm;
    REAL_T *d_einput_data, *d_eoutput_data;
    REAL_T *d_ninput_data, *d_noutput_data;
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_Y, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_error, sizeof(REAL_T));
    cudaMalloc((void **) &d_norm, sizeof(REAL_T));

    cudaMalloc((void **) &d_einput_data, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_ninput_data, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_eoutput_data, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_noutput_data, sizeof(REAL_T) * n);
    printf("done\n");
    // transfer data
    printf("Copying data into device... ");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeof(REAL_T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(REAL_T) * n, cudaMemcpyHostToDevice);
    norm = 0; // reset norm
    printf("done\n");

    cudaDeviceSynchronize();

    start_time = my_gettimeofday();
    {
        n_iterations = 0;
        error = INFINITY;

        printf("GridSize: %d, BlockSize: %d\n", gridSize.x, blockSize.x);
        while (error > ERROR_THRESHOLD) {
            // printf("Itération %4d -- Error: %g\n", n_iterations, error);

            {
                profile_cuda_scope("vec mat mult kernel");
                vecMatMultKernel<<<gridSize, blockSize>>>(d_A, d_X, d_Y, n);
            }

            // norm
            {
                profile_cuda_scope("norm sum kernel");
                // square and output in d_output_data (could be its own kernel with reduction)
                normSumKernel<<<gridSize, blockSize>>>(d_Y, n, d_ninput_data);
                reduceArray(d_ninput_data, d_noutput_data, n);
            }

            {
                profile_cuda_scope("normalize y");
                normalizeYKernel<<<gridSize, blockSize>>>(d_Y, &d_noutput_data[0], n);
            }

            // calculate error
            {
                profile_cuda_scope("error kernel");
                errorKernel<<<gridSize, blockSize>>>(d_Y, d_X, n, d_einput_data);
                reduceArray(d_einput_data, d_eoutput_data, n);
                cudaMemcpy(&error, d_eoutput_data, sizeof(REAL_T), cudaMemcpyDeviceToHost);
                error = sqrt(error);
            }

            // swap device pointers
            REAL_T *d_tmp = d_X;
            d_X = d_Y;
            d_Y = d_tmp;

            n_iterations++;
            if(error == INFINITY) {
                printf("Error happened\n");
                break;
            }
        }
        // get back eigen vector and norm
        cudaMemcpy(&norm, d_noutput_data, sizeof(REAL_T), cudaMemcpyDeviceToHost);
        norm = sqrt(norm);
        cudaMemcpy(X, d_X, sizeof(REAL_T) * n, cudaMemcpyDeviceToHost);
    }
    total_time = my_gettimeofday() - start_time;
    printf("final error after %4d iterations: %g (|VP| = %g)\n", n_iterations, error, norm);
    printf("time: %.1f s      Mflop/s: %.1f \n", total_time,
           (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);


    /*** Storing the eigen vector in a file ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("Unable to open result.out in write mode");
        exit(1);
    }
    fprintf(output, "%d\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", X[i]);
    }
    fclose(output);

    free(A);
    cudaFree(d_A);
    free(X);
    cudaFree(d_X);
    free(Y);
    cudaFree(d_Y);
    cudaFree(d_einput_data);
    cudaFree(d_eoutput_data);
    cudaFree(d_noutput_data);
    cudaFree(d_noutput_data);
}
