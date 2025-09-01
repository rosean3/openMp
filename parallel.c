#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

#define N_POINTS 10
#define N_ANTS 57
#define N_ITER 1000

// Parameters
#define ALPHA 0.3382068189986327
#define BETA 0.9883984716309214
#define EVAPORATION_RATE 0.540492617385571
#define Q 2.8457919863458874

// Example points
double points[N_POINTS][3] = {
    {0.12006853, 0.58435911, 0.0211234},  {0.08637881, 0.50941081, 0.6076127},
    {0.11863504, 0.16507902, 0.62409166}, {0.07216265, 0.78416184, 0.57923631},
    {0.20443071, 0.65579228, 0.10723222}, {0.10297309, 0.00728977, 0.33962577},
    {0.27886254, 0.86811759, 0.22386256}, {0.34442165, 0.2938858, 0.56934684},
    {0.51967557, 0.66116049, 0.22149384}, {0.61048266, 0.30626978, 0.32945194}};

double distance(int i, int j) {
  double dx = points[i][0] - points[j][0];
  double dy = points[i][1] - points[j][1];
  double dz = points[i][2] - points[j][2];
  return sqrt(dx * dx + dy * dy + dz * dz);
}

int roulette_wheel(int n, double *prob) {
  double r = ((double)rand() / RAND_MAX);
  double cumulative = 0.0;
  for (int i = 0; i < n; i++) {
    cumulative += prob[i];
    if (r <= cumulative)
      return i;
  }
  return n - 1;
}

void visualize_path(int *best_path) {
  FILE *fp = fopen("path.dat", "w");
  if (!fp) {
    perror("Error writing path.dat");
    return;
  }

  // Write all points
  fprintf(fp, "# All points\n");
  for (int i = 0; i < N_POINTS; i++) {
    fprintf(fp, "%f %f %f\n", points[i][0], points[i][1], points[i][2]);
  }
  fprintf(fp, "\n\n");

  // Write path edges
  fprintf(fp, "# Path\n");
  for (int i = 0; i < N_POINTS; i++) {
    int j = (i + 1) % N_POINTS;
    fprintf(fp, "%f %f %f\n", points[best_path[i]][0], points[best_path[i]][1],
            points[best_path[i]][2]);
    fprintf(fp, "%f %f %f\n\n", points[best_path[j]][0],
            points[best_path[j]][1], points[best_path[j]][2]);
  }

  fclose(fp);

  // Launch gnuplot
  FILE *gp = popen("gnuplot -persistent", "w");
  if (gp == NULL) {
    perror("Error opening gnuplot");
    return;
  }

  fprintf(gp, "set title 'Ant Colony Optimization Path'\n");
  fprintf(gp, "set xlabel 'X'\n");
  fprintf(gp, "set ylabel 'Y'\n");
  fprintf(gp, "set zlabel 'Z'\n");

  // Plot: first point in blue, rest points in red, path in green
  fprintf(gp, "splot '-' using 1:2:3 with points pointtype 7 pointsize 2 lc "
              "rgb 'blue' title 'Start Point',\\\n");
  fprintf(gp, "      '-' using 1:2:3 with points pointtype 7 pointsize 1 lc "
              "rgb 'red' title 'Other Points',\\\n");
  fprintf(gp, "      '-' using 1:2:3 with lines lc rgb 'green' lw 2 title "
              "'Best Path'\n");

  // Start point
  fprintf(gp, "%f %f %f\n", points[0][0], points[0][1], points[0][2]);
  fprintf(gp, "e\n");

  // Other points (excluding start)
  for (int i = 1; i < N_POINTS; i++)
    fprintf(gp, "%f %f %f\n", points[i][0], points[i][1], points[i][2]);
  fprintf(gp, "e\n");

  // Path edges
  for (int i = 0; i < N_POINTS; i++) {
    int j = (i + 1) % N_POINTS;
    fprintf(gp, "%f %f %f\n", points[best_path[i]][0], points[best_path[i]][1],
            points[best_path[i]][2]);
    fprintf(gp, "%f %f %f\n\n", points[best_path[j]][0],
            points[best_path[j]][1], points[best_path[j]][2]);
  }
  fprintf(gp, "e\n");

  fflush(gp);
}

int main() {
  srand(time(NULL));

  double pheromone[N_POINTS][N_POINTS];
  for (int i = 0; i < N_POINTS; i++)
    for (int j = 0; j < N_POINTS; j++)
      pheromone[i][j] = 1.0;

  double best_length = 1e9;
  int best_path[N_POINTS];

  for (int iter = 0; iter < N_ITER; iter++) {
    // temporary pheromone matrix for this iteration
    double delta_pheromone[N_POINTS][N_POINTS] = {0};

#pragma omp parallel for schedule(dynamic)
    for (int ant = 0; ant < N_ANTS; ant++) {
      int visited[N_POINTS] = {0};
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
          double eta = pow(1.0 / distance(current, next), BETA);
          prob[k] = tau * eta;
          sum += prob[k];
        }
        for (int k = 0; k < n_unvisited; k++)
          prob[k] /= sum;

        int idx = roulette_wheel(n_unvisited, prob);
        int next_point = unvisited[idx];

        length += distance(current, next_point);
        current = next_point;
        path[step] = current;
        visited[current] = 1;
      }

// update best path (critical section)
#pragma omp critical
      {
        if (length < best_length) {
          best_length = length;
          for (int i = 0; i < N_POINTS; i++)
            best_path[i] = path[i];
        }
      }

// update local pheromone
#pragma omp critical
      {
        for (int i = 0; i < N_POINTS - 1; i++) {
          delta_pheromone[path[i]][path[i + 1]] += Q / length;
        }
        delta_pheromone[path[N_POINTS - 1]][path[0]] += Q / length;
      }
    }

    // evaporate + deposit pheromone
    for (int i = 0; i < N_POINTS; i++) {
      for (int j = 0; j < N_POINTS; j++) {
        pheromone[i][j] *= EVAPORATION_RATE;
        pheromone[i][j] += delta_pheromone[i][j];
      }
    }
  }

  // Print best result
  printf("Best path length: %f\n", best_length);
  printf("Best path: ");
  for (int i = 0; i < N_POINTS; i++)
    printf("%d ", best_path[i]);
  printf("\n");

  return 0;
}
