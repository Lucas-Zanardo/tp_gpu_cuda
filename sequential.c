/**
 * HPC - M2 Data Science - Univ. Lille 
 * Authors: C. Bouillaguet and P. Fortin 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

int main(int argc, char **argv) {
    int i, j, n;
    long long size;
    REAL_T norm, inv_norm, error, delta;
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
    printf("Matrix size: %.3f G\n", size / 1073741824.);

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
    for (i = 0; i < n; i++) {
        init_row(A, i, n);
    }

    for (i = 0; i < n; i++) {
        X[i] = 1.0 / n;
    }

    start_time = my_gettimeofday();
    error = INFINITY;
    n_iterations = 0;
    while (error > ERROR_THRESHOLD) {
        // printf("iteration %4d, current error %g\n", n_iterations, error);

        /*** y <--- A.x ***/
        for (i = 0; i < n; i++) {
            Y[i] = 0;
            for (j = 0; j < n; j++) {
                Y[i] += A[i * n + j] * X[j];
            }
        }

        /*** norm <--- ||y|| ***/
        norm = 0;
        for (i = 0; i < n; i++) {
            norm += Y[i] * Y[i];
        }
        norm = sqrt(norm);

        /*** y <--- y / ||y|| ***/
        inv_norm = 1.0 / norm;
        for (i = 0; i < n; i++) {
            Y[i] *= inv_norm;
        }

        /*** error <--- ||x - y|| ***/
        error = 0;
        for (i = 0; i < n; i++) {
            delta = X[i] - Y[i];
            error += delta * delta;
        }
        error = sqrt(error);

        /*** x <--> y ***/
        REAL_T *tmp = X;
        X = Y;
        Y = tmp;

        n_iterations++;
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
    free(X);
    free(Y);
}
