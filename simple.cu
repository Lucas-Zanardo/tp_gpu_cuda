/**
 * HPC - M2 Data Science - Univ. Lille
 * Authors: C. Bouillaguet and P. Fortin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef ELEMENTS_PER_BLOCK
#define ELEMENTS_PER_BLOCK 128
#endif

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
        REAL_T *norm,
        int n,
        REAL_T *odata
) {
    extern __shared__ REAL_T sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = Y[i];
    }
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) odata[blockIdx.x] = sdata[0];
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
        REAL_T *error,
        int n
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        REAL_T delta = X[i] - Y[i];
        atomicAdd(error, delta * delta);
    }
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
    // TODO: Kernel pour initialisation des matrices
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
    printf("Allocating device memory\n");
    REAL_T *d_A, *d_X, *d_Y;
    REAL_T *d_error, *d_norm;
    REAL_T *d_norm_tab, *norm_tab;
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_Y, sizeof(REAL_T) * n);
    cudaMalloc((void **) &d_error, sizeof(REAL_T));
    cudaMalloc((void **) &d_norm, sizeof(REAL_T));

    norm_tab = (REAL_T *) malloc(sizeof(REAL_T) * gridSize.x);
    cudaMalloc((void **) &d_norm_tab, sizeof(REAL_T) * gridSize.x);
    printf("Allocated device memory\n");
    // transfer data
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeof(REAL_T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(REAL_T) * n, cudaMemcpyHostToDevice);
    norm = 0; // reset norm
    printf("Copied data into device\n");


    start_time = my_gettimeofday();
    {
        n_iterations = 0;
        error = INFINITY;

        printf("GridSize: %d, BlockSize: %d\n", gridSize.x, blockSize.x);
        while (error > ERROR_THRESHOLD) {
            vecMatMultKernel<<<gridSize, blockSize>>>(d_A, d_X, d_Y, n);

            // norm
            normSumKernel<<<gridSize, blockSize, sizeof(REAL_T) * ELEMENTS_PER_BLOCK>>>(d_Y, d_norm, n, d_norm_tab);
            // Add norm tab on cpu side
            cudaMemcpy(norm_tab, d_norm_tab, sizeof(REAL_T) * gridSize.x, cudaMemcpyDeviceToHost);
            norm = 0;
            for (i = 0; i < gridSize.x; ++i) {
                norm += norm_tab[i];
            }
            cudaMemcpy(d_norm, &norm, sizeof(REAL_T), cudaMemcpyHostToDevice);
            normalizeYKernel<<<gridSize, blockSize>>>(d_Y, d_norm, n);

            // calculate error
            error = 0;
            cudaMemcpy(d_error, &error, sizeof(REAL_T), cudaMemcpyHostToDevice);
            errorKernel<<<gridSize, blockSize>>>(d_Y, d_X, d_error, n);
            cudaMemcpy(&error, d_error, sizeof(REAL_T), cudaMemcpyDeviceToHost);
            error = sqrt(error);

            // swap device pointers
            REAL_T *d_tmp = d_X;
            d_X = d_Y;
            d_Y = d_tmp;

            n_iterations++;
        }
        // get back eigen vector and norm
        cudaMemcpy(&norm, d_norm, sizeof(REAL_T), cudaMemcpyDeviceToHost);
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
}
