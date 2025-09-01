#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Parameters
#define N_ANTS 57
#define ALPHA 0.3382068189986327
#define BETA 0.9883984716309214
#define EVAPORATION_RATE 0.540492617385571
#define Q 2.8457919863458874

int main() {
  srand(time(NULL));

  int N_POINTS = 0;
  printf("Enter the number of points: ");
  if (scanf("%d", &N_POINTS) != 1) {
    return 1;
  }

  int N_ITER = 10000;
  printf("Enter the number of iterations: ");
  if (scanf("%d", &N_ITER) != 1) {
    return 1;
  }

  int visualize = 0;
  printf("Do you wish to visualize the result (0/1)?: ");
  if (scanf("%d", &visualize) != 1) {
    return 1;
  }

  double points[N_POINTS][3];
  int n_points = read_points("points.txt", points, N_POINTS);

  // Start timing
  clock_t start_time = clock();

  double pheromone[N_POINTS][N_POINTS];
  for (int i = 0; i < N_POINTS; i++)
    for (int j = 0; j < N_POINTS; j++)
      pheromone[i][j] = 1.0;

  double global_best_length = 1e9;
  int global_best_path[N_POINTS];

  for (int iter = 0; iter < N_ITER; iter++) {
    int n_threads;
#pragma omp parallel
    { n_threads = omp_get_num_threads(); }

    // thread-local pheromone updates
    double delta_pheromone_private[n_threads][N_POINTS][N_POINTS];
    for (int t = 0; t < n_threads; t++)
      for (int i = 0; i < N_POINTS; i++)
        for (int j = 0; j < N_POINTS; j++)
          delta_pheromone_private[t][i][j] = 0.0;

    // thread-local bests
    double local_best_length[n_threads];
    int local_best_path[n_threads][N_POINTS];
    for (int t = 0; t < n_threads; t++)
      local_best_length[t] = 1e9;

// parallel ant loop
#pragma omp parallel for schedule(dynamic)
    for (int ant = 0; ant < N_ANTS; ant++) {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned int)time(NULL) ^ (tid + ant * 31);

      int visited[N_POINTS];
      for (int v = 0; v < N_POINTS; v++)
        visited[v] = 0;
      int path[N_POINTS];
      int current = 0;
      path[0] = current;
      visited[current] = 1;
      double length = 0.0;

      // Build path
      for (int step = 1; step < N_POINTS; step++) {
        int unvisited[N_POINTS];
        int n_unvisited = 0;
        for (int k = 0; k < N_POINTS; k++) {
          if (!visited[k]) {
            unvisited[n_unvisited++] = k;
          }
        }

        double prob[n_unvisited];
        double sum = 0.0;
        for (int k = 0; k < n_unvisited; k++) {
          int next = unvisited[k];
          double tau = pow(pheromone[current][next], ALPHA);
          double eta = pow(1.0 / distance(current, next, points), BETA);
          prob[k] = tau * eta;
          sum += prob[k];
        }
        for (int k = 0; k < n_unvisited; k++)
          prob[k] /= sum;

        int idx = roulette_wheel(n_unvisited, prob);
        int next_point = unvisited[idx];

        length += distance(current, next_point, points);
        current = next_point;
        path[step] = current;
        visited[current] = 1;
      }

      // local best update
      if (length < local_best_length[tid]) {
        local_best_length[tid] = length;
        for (int i = 0; i < N_POINTS; i++)
          local_best_path[tid][i] = path[i];
      }

      // update thread-local pheromone
      for (int i = 0; i < N_POINTS - 1; i++) {
        delta_pheromone_private[tid][path[i]][path[i + 1]] += Q / length;
      }
      delta_pheromone_private[tid][path[N_POINTS - 1]][path[0]] += Q / length;
    }

    // merge pheromone contributions
    double delta_pheromone[N_POINTS][N_POINTS];
    for (int t = 0; t < n_threads; t++) {
      for (int i = 0; i < N_POINTS; i++) {
        for (int j = 0; j < N_POINTS; j++) {
          if (t == 0) { // initialize to 0
              delta_pheromone[i][j] = 0;
          }
          delta_pheromone[i][j] += delta_pheromone_private[t][i][j];
        }
      }
    }

    // evaporate + deposit pheromone
    for (int i = 0; i < N_POINTS; i++) {
      for (int j = 0; j < N_POINTS; j++) {
        pheromone[i][j] *= EVAPORATION_RATE;
        pheromone[i][j] += delta_pheromone[i][j];
      }
    }

    // merge best paths
    for (int t = 0; t < n_threads; t++) {
      if (local_best_length[t] < global_best_length) {
        global_best_length = local_best_length[t];
        for (int i = 0; i < N_POINTS; i++)
          global_best_path[i] = local_best_path[t][i];
      }
    }
  }

  clock_t end_time = clock();
  double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  // Print best result
  printf("Best path length: %f\n", global_best_length);
  printf("Best path: ");
  for (int i = 0; i < N_POINTS; i++)
    printf("%d ", global_best_path[i]);
  printf("\n");

  printf("Elapsed time: %.3f seconds\n", elapsed_sec);

  return 0;
}

// gcc parallel2.c utils.c -o parallel2 -lm -fopenmp