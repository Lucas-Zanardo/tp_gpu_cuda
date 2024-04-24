#ifndef DEFS_H
#define DEFS_H

///// double precision: 
//#define REAL_T double 
//#define ERROR_THRESHOLD 1e-9

///// single precision:
#define REAL_T float
#define ERROR_THRESHOLD 1e-5

///// Random number generator: 
#define PRNG_MAX 0x0007FFFFFFFll
#define PRNG_1   0x00473EE661Dll
#define PRNG_2   0x024719D0275ll
#define RANGE    101

///// Timing: 
static double my_gettimeofday() {
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

// Initialize the row #i, with n elements, of the A matrix: 
static void init_row(REAL_T *A, long i, long n) {
    for (long j = 0; j < n; j++) {
        A[i * n + j] = (((REAL_T) ((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
    }
    for (long k = 1; k < n; k *= 2) {
        if (i + k < n) {
            A[i * n + i + k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
        }
        if (i - k >= 0) {
            A[i * n + i - k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
        }
    }
}

#endif //DEFS_H