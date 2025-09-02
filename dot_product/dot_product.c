#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main() {
  char parallel_or_sequential = 0;
  printf("Do you wish to run the parallel or sequential version (p/s)?: ");
  if (scanf(" %c", &parallel_or_sequential) != 1 || (parallel_or_sequential != 'p' && parallel_or_sequential != 's'))
    return 1;

  double start_time = omp_get_wtime();

  size_t N = 100000000;

  double dot_product = 0.0;
  
  double *A = aligned_alloc(64, N * sizeof(double));

  double *B = aligned_alloc(64, N * sizeof(double));

  // inicialização - não vale a pena paralelizar
  for (size_t i = 0; i < N; i++) { A[i] = 444444; B[i] = 333333; };

  if (parallel_or_sequential == 'p') {
    #pragma omp parallel for reduction(+:dot_product) schedule(static)
    for (size_t i = 0; i < N; i++) {
        dot_product += A[i] * B[i];
    }
  } else if (parallel_or_sequential == 's') {
      for (size_t i = 0; i < N; i++) {
          dot_product += A[i] * B[i];
      }
  }

  double end_time = omp_get_wtime();
  printf("The dot product is: %f\n", dot_product);
  printf("Elapsed time: %.3f seconds\n", end_time - start_time);
  
  return 0;
}