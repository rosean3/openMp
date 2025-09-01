#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include "config.h"

int main() {
  srand(time(NULL));

  int num_points = 0;
  printf("Enter the number of points: ");
  if (scanf("%d", &num_points) != 1) {
    return 1;
  }

  int n_iterations = 10000;
  printf("Enter the number of iterations: ");
  if (scanf("%d", &n_iterations) != 1) {
    return 1;
  }

  int visualize = 0;
  printf("Do you wish to visualize the result (0/1)?: ");
  if (scanf("%d", &visualize) != 1) {
    return 1;
  }

  char parallel_or_sequential = 0;
  printf("Do you wish to run the parallel or sequential version (p/s)?: ");
  if (scanf(" %c", &parallel_or_sequential) != 1)
    return 1;

  double points[num_points][3];
  int n_points = read_points("points.txt", points, num_points);

  // Start timing
  clock_t start_time = clock();

  ACOResult result;
  
  if (parallel_or_sequential == 'p') {
    result = parallel_ant_colony_optimization(points, num_points, n_iterations);
  } else if (parallel_or_sequential == 's') {
    result = sequential_ant_colony_optimization(points, num_points, n_iterations);
  }

  clock_t end_time = clock();
  double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  // Print best path
  printf("Best path length: %f\n", result.best_length);
  printf("Best path: ");
  for (int i = 0; i < num_points; i++)
    printf("%d ", result.best_path[i]);
  printf("\n");

  printf("Elapsed time: %.3f seconds\n", elapsed_sec);

  if (visualize == 1) {
    visualize_path(num_points, points, result.best_path);
  }

  return 0;
}

// gcc aco.c utils.c -o aco -lm -fopenmp