/**
 * HPC - M2 Data Science - Univ. Lille 
 * Authors: C. Bouillaguet and P. Fortin  
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "defs.h"

int main(int argc, char **argv){
  long i, j, k, n;
  REAL_T norm, inv_norm, error, delta;
  double start_time;
  REAL_T *A, *X, *Y;
  FILE *input;

  if (argc < 2) {
    printf("USAGE: %s [file]\n", argv[0]);
    exit(1);
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Unable to open %s in read mode", argv[1]);
    exit(1);
  }
  fscanf(input, "%ld", &n);
  printf("Read: n = %ld\n", n);


  X = (REAL_T *)malloc(n * sizeof(REAL_T));
  Y = (REAL_T *)malloc(n * sizeof(REAL_T));
  if ((X == NULL) || (Y == NULL)) {
    perror("Unable to allocate memory");
    exit(1);
  }
  for (i = 0; i < n; i++) {
    fscanf(input, (sizeof(REAL_T) == sizeof(double) ? "%lf\n" : "%f\n"), X + i);
  }
  fclose(input);

  start_time = my_gettimeofday();
  error = 0;

  A = (REAL_T *)malloc(n * n * sizeof(REAL_T));
  if (A == NULL) {
    perror("Unable to allocate memory");
    exit(1);
  }
  
  for (i = 0; i < n; i++) {
    init_row(A, i, n);
  }

  for (i = 0; i < n; i++) {
    Y[i] = 0;
    for (j = 0; j < n; j++) {
      Y[i] += A[i*n+j] * X[j];
    }
  }

  free(A);
  

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

  printf("Error: %g (|VP| = %g)\n", error, norm);
  printf("Time: %.1f s\n", my_gettimeofday() - start_time);

  free(X);
  free(Y);
}
