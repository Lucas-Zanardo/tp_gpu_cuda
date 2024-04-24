#ifdef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda.h>
#include <stdio.h>


/** 
 * CUDA error handling and debugging 
 */

#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR() {						\
    cudaError_t sync_error;						\
    cudaDeviceSynchronize();						\
    sync_error = cudaGetLastError();					\
    if(sync_error != cudaSuccess) {					\
      fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",		\
        __FILE__ , __LINE__, cudaGetErrorString(sync_error));	\
      exit(EXIT_FAILURE);						\
    }									\
  }
#else /* #ifdef CUDA_DEBUG */
#define CUDA_SYNC_ERROR()
#endif /* #ifdef CUDA_DEBUG */

#define CUDA_ERROR(cuda_call) {					\
    cudaError_t error = cuda_call;				\
    if(error != cudaSuccess){					\
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",		\
        __FILE__ , __LINE__, cudaGetErrorString(error));	\
      exit(EXIT_FAILURE);					\
    }								\
    CUDA_SYNC_ERROR();						\
  }



/**
 * Returns the integer quotient greater or equal to "a/b". 
 * From: CUDA SDK 4.1 
 */

static int iDivUp(int a, int b){
  return ((a % b != 0) ? (a / b + 1) : (a / b));
}

#endif // CUDA_COMMON_H