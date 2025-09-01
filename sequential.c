#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

  double best_length = 1e9;
  int best_path[N_POINTS];

  for (int iter = 0; iter < N_ITER; iter++) {
    for (int ant = 0; ant < N_ANTS; ant++) {
      int visited[N_POINTS];
      for (int v = 0; v < N_POINTS; v++)
        visited[v] = 0;

      int path[N_POINTS];
      int current = 0;
      path[0] = current;
      visited[current] = 1;
      double length = 0.0;

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

      // Update best path
      if (length < best_length) {
        best_length = length;
        for (int i = 0; i < N_POINTS; i++)
          best_path[i] = path[i];
      }

      // Pheromone update
      for (int i = 0; i < N_POINTS - 1; i++) {
        pheromone[path[i]][path[i + 1]] += Q / length;
      }
      pheromone[path[N_POINTS - 1]][path[0]] += Q / length;
    }

    // Evaporation
    for (int i = 0; i < N_POINTS; i++)
      for (int j = 0; j < N_POINTS; j++)
        pheromone[i][j] *= EVAPORATION_RATE;
  }

  clock_t end_time = clock();
  double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  // Print best path
  printf("Best path length: %f\n", best_length);
  printf("Best path: ");
  for (int i = 0; i < N_POINTS; i++)
    printf("%d ", best_path[i]);
  printf("\n");

  printf("Elapsed time: %.3f seconds\n", elapsed_sec);

  if (visualize == 1) {
    visualize_path(N_POINTS, points, best_path);
  }

  return 0;
}

// gcc sequential.c utils.c -o sequential -lm