#include "utils.h"
#include <math.h>
#include <time.h>
#include "config.h"
#include <omp.h>

double distance(int i, int j, double points[][3]) {
  double dx = points[i][0] - points[j][0];
  double dy = points[i][1] - points[j][1];
  double dz = points[i][2] - points[j][2];
  return sqrt(dx * dx + dy * dy + dz * dz);
}

int roulette_wheel(int n, double *probabilities) {
  double r = (double)rand() / RAND_MAX;
  double cumulative = 0.0;
  for (int i = 0; i < n; i++) {
    cumulative += probabilities[i];
    if (r <= cumulative)
      return i;
  }
  return n - 1; // fallback
}

void visualize_path(int N_POINTS, double points[][3], int *best_path) {
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

// Reads 3D points from file, returns number of points read
int read_points(const char *filename, double points[][3], int max_points) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: could not open %s\n", filename);
    return -1;
  }

  int count = 0;
  while (count < max_points &&
         fscanf(fp, "%lf %lf %lf", &points[count][0], &points[count][1],
                &points[count][2]) == 3) {
    count++;
  }

  fclose(fp);
  return count;
}

ACOResult parallel_ant_colony_optimization(double points[][3], int N_POINTS, int N_ITER) {
  printf("Parallel ACO\n");
  
  double best_length = 1e9;
  int best_path[N_POINTS];
  
  double pheromone[N_POINTS][N_POINTS];
    for (int i = 0; i < N_POINTS; i++)
      for (int j = 0; j < N_POINTS; j++)
        pheromone[i][j] = 1.0;

    for (int iter = 0; iter < N_ITER; iter++) {
      // temporary pheromone matrix for this iteration
      double delta_pheromone[N_POINTS][N_POINTS];
      for (int i = 0; i < N_POINTS; i++)
        for (int j = 0; j < N_POINTS; j++)
          delta_pheromone[i][j] = 0.0;
      

  #pragma omp parallel for schedule(dynamic)
      for (int ant = 0; ant < N_ANTS; ant++) {
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

  ACOResult result = {best_length, best_path};
  return result;
}

ACOResult sequential_ant_colony_optimization(double points[][3], int N_POINTS, int N_ITER) {
  printf("Sequential ACO\n");
  
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

  ACOResult result = {best_length, best_path};
  return result;
}

ACOResult parallel2_ant_colony_optimization(double points[][3], int N_POINTS, int N_ITER) {
  printf("Parallel2 ACO\n");
  
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

  ACOResult result = {global_best_length, global_best_path};
  return result;
}