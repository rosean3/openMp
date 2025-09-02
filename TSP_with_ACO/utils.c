#include "utils.h"
#include <math.h>
#include <time.h>
#include "config.h"
#include <omp.h>
#include <string.h>

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
    fprintf(fp, "%f %f %f\n", points[best_path[i]][0], points[best_path[i]][1], points[best_path[i]][2]);
    fprintf(fp, "%f %f %f\n\n", points[best_path[j]][0], points[best_path[j]][1], points[best_path[j]][2]);
  }

  fclose(fp);

  // Launch gnuplot (output to PNG instead of X11)
  FILE *gp = popen("gnuplot", "w");
  if (gp == NULL) {
    perror("Error opening gnuplot");
    return;
  }

  fprintf(gp, "set terminal pngcairo size 800,600\n");
  fprintf(gp, "set output 'path.png'\n");
  fprintf(gp, "set title 'Ant Colony Optimization Path'\n");
  fprintf(gp, "set xlabel 'X'\n");
  fprintf(gp, "set ylabel 'Y'\n");
  fprintf(gp, "set zlabel 'Z'\n");

  // Plot: first point blue, rest red, path green
  fprintf(gp, "splot '-' using 1:2:3 with points pointtype 7 pointsize 2 lc rgb 'blue' title 'Start Point',\\\n");
  fprintf(gp, "      '-' using 1:2:3 with points pointtype 7 pointsize 1 lc rgb 'red' title 'Other Points',\\\n");
  fprintf(gp, "      '-' using 1:2:3 with lines lc rgb 'green' lw 2 title 'Best Path'\n");

  // Start point
  fprintf(gp, "%f %f %f\n", points[0][0], points[0][1], points[0][2]);
  fprintf(gp, "e\n");

  // Other points
  for (int i = 1; i < N_POINTS; i++)
    fprintf(gp, "%f %f %f\n", points[i][0], points[i][1], points[i][2]);
  fprintf(gp, "e\n");

  // Path edges
  for (int i = 0; i < N_POINTS; i++) {
    int j = (i + 1) % N_POINTS;
    fprintf(gp, "%f %f %f\n", points[best_path[i]][0], points[best_path[i]][1], points[best_path[i]][2]);
    fprintf(gp, "%f %f %f\n\n", points[best_path[j]][0], points[best_path[j]][1], points[best_path[j]][2]);
  }
  fprintf(gp, "e\n");

  fflush(gp);
  pclose(gp);
}


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

ACOResult sequential_ant_colony_optimization(double points[][3], int N_POINTS, int N_ITER) {
  printf("Sequential ACO\n");

  size_t NN = (size_t)N_POINTS * (size_t)N_POINTS;
  double *pheromone = malloc(NN * sizeof(double));
  if (!pheromone) { fprintf(stderr, "Allocation failed (pheromone)\n"); exit(EXIT_FAILURE); }

  for (size_t k = 0; k < NN; k++) pheromone[k] = 1.0;

  double best_length = 1e9;
  int *best_path = malloc(N_POINTS * sizeof(int));
  if (!best_path) { fprintf(stderr, "Allocation failed (best_path)\n"); exit(EXIT_FAILURE); }

  for (int iter = 0; iter < N_ITER; iter++) {
    for (int ant = 0; ant < N_ANTS; ant++) {
      // put temporaries on heap instead of alloca (safer for big N_POINTS)
      int *visited   = calloc(N_POINTS, sizeof(int));
      int *path      = malloc(N_POINTS * sizeof(int));
      int *unvisited = malloc(N_POINTS * sizeof(int));
      double *prob   = malloc(N_POINTS * sizeof(double));
      if (!visited || !path || !unvisited || !prob) {
        fprintf(stderr, "Allocation failed (temporary arrays)\n");
        exit(EXIT_FAILURE);
      }

      int current = 0;
      path[0] = current;
      visited[current] = 1;
      double length = 0.0;

      for (int step = 1; step < N_POINTS; step++) {
        int n_unvisited = 0;
        for (int k = 0; k < N_POINTS; k++)
          if (!visited[k]) unvisited[n_unvisited++] = k;

        double sum = 0.0;
        for (int k = 0; k < n_unvisited; k++) {
          int next = unvisited[k];
          double d = distance(current, next, points);
          if (d <= 1e-12) d = 1e-12;  // prevent division by zero
          double tau = pow(pheromone[IDX(current, next, N_POINTS)], ALPHA);
          double eta = pow(1.0 / d, BETA);
          prob[k] = tau * eta;
          sum += prob[k];
        }

        // normalize probabilities safely
        if (sum <= 1e-12) {
          for (int k = 0; k < n_unvisited; k++) prob[k] = 1.0 / n_unvisited;
        } else {
          for (int k = 0; k < n_unvisited; k++) prob[k] /= sum;
        }

        int idx = roulette_wheel(n_unvisited, prob);
        if (idx < 0 || idx >= n_unvisited) idx = 0;  // guard bad return

        int next_point = unvisited[idx];
        length += distance(current, next_point, points);
        current = next_point;
        path[step] = current;
        visited[current] = 1;
      }

      // Update best path
      if (length < best_length) {
        best_length = length;
        memcpy(best_path, path, sizeof(int) * (size_t)N_POINTS);
      }

      // Pheromone update
      for (int i = 0; i < N_POINTS - 1; i++) {
        size_t a = (size_t)path[i];
        size_t b = (size_t)path[i + 1];
        pheromone[IDX(a, b, N_POINTS)] += Q / length;
      }
      pheromone[IDX(path[N_POINTS - 1], path[0], N_POINTS)] += Q / length;

      free(visited);
      free(path);
      free(unvisited);
      free(prob);
    }

    // Evaporation
    for (size_t k = 0; k < NN; k++) pheromone[k] *= EVAPORATION_RATE;
  }

  free(pheromone);

  ACOResult result = { best_length, best_path };
  return result;
}

typedef struct {
  double length;
  char pad[CACHELINE - sizeof(double)];
} AlignedDouble;

ACOResult parallel_ant_colony_optimization(double points[][3], int N_POINTS, int N_ITER) {
  printf("Manual batch ACO\n");

  // Decide threads up front
  int n_threads = omp_get_max_threads();

  // Allocate big things on the heap
  size_t NN = (size_t)N_POINTS * (size_t)N_POINTS;

  double *pheromone = (double*)malloc(NN * sizeof(double));
  double *delta_pheromone = (double*)malloc(NN * sizeof(double));
  // Per-thread delta: [n_threads][N_POINTS][N_POINTS] flattened
  double *delta_pheromone_private = (double*)calloc((size_t)n_threads * NN, sizeof(double));
  if (!pheromone || !delta_pheromone || !delta_pheromone_private) {
    fprintf(stderr, "Allocation failed (pheromone arrays)\n");
    exit(EXIT_FAILURE);
  }

  // Init pheromone = 1.0
  for (size_t k = 0; k < NN; ++k) pheromone[k] = 1.0;

  // Bests
  double global_best_length = 1e9;
  int* global_best_path = (int*)malloc(sizeof(int) * (size_t)N_POINTS);
  if (!global_best_path) {
    fprintf(stderr, "Allocation failed (global_best_path)\n");
    exit(EXIT_FAILURE);
  }

  // Local best buffers
  AlignedDouble *local_best_length = (AlignedDouble*)malloc(sizeof(AlignedDouble) * (size_t)n_threads);
  int *local_best_path = (int*)malloc(sizeof(int) * (size_t)n_threads * (size_t)N_POINTS);
  if (!local_best_length || !local_best_path) {
    fprintf(stderr, "Allocation failed (local bests)\n");
    exit(EXIT_FAILURE);
  }

  for (int t = 0; t < n_threads; ++t) {
    local_best_length[t].length = 1e9;
  }

  for (int iter = 0; iter < N_ITER; iter++) {
    // Clear per-thread deltas
    memset(delta_pheromone_private, 0, (size_t)n_threads * NN * sizeof(double));
    for (int t = 0; t < n_threads; ++t) local_best_length[t].length = 1e9;

    // Manual ant batching
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int ants_per_thread = (N_ANTS + n_threads - 1) / n_threads;  // ceiling
      int start_ant = tid * ants_per_thread;
      int end_ant = (start_ant + ants_per_thread > N_ANTS) ? N_ANTS : start_ant + ants_per_thread;

      unsigned int seed = (unsigned int)time(NULL) ^ (tid * 2654435761U);

      // Per-thread view into its delta slice
      double *dpriv = delta_pheromone_private + (size_t)tid * NN;

      for (int ant = start_ant; ant < end_ant; ant++) {
        (void)seed; // in case your roulette uses an external RNG

        int *visited = (int*)alloca(sizeof(int) * (size_t)N_POINTS);
        int *path    = (int*)alloca(sizeof(int) * (size_t)N_POINTS);
        int *unvisited = (int*)alloca(sizeof(int) * (size_t)N_POINTS);
        double *prob = (double*)alloca(sizeof(double) * (size_t)N_POINTS);

        memset(visited, 0, sizeof(int) * (size_t)N_POINTS);

        int current = 0;
        path[0] = current;
        visited[current] = 1;
        double length = 0.0;

        for (int step = 1; step < N_POINTS; step++) {
          int n_unvisited = 0;
          for (int k = 0; k < N_POINTS; k++) {
            if (!visited[k]) unvisited[n_unvisited++] = k;
          }

          double sum = 0.0;
          for (int k = 0; k < n_unvisited; k++) {
            int next = unvisited[k];
            double tau = pow(pheromone[IDX(current, next, N_POINTS)], ALPHA);
            double eta = pow(1.0 / distance(current, next, points), BETA);
            prob[k] = tau * eta;
            sum += prob[k];
          }

          // Guard against degenerate sums
          if (sum <= 0.0) {
            for (int k = 0; k < n_unvisited; k++) prob[k] = 1.0 / n_unvisited;
          } else {
            for (int k = 0; k < n_unvisited; k++) prob[k] /= sum;
          }

          int idx = roulette_wheel(n_unvisited, prob);
          int next_point = unvisited[idx];

          length += distance(current, next_point, points);
          current = next_point;
          path[step] = current;
          visited[current] = 1;
        }

        // Update local best
        if (length < local_best_length[tid].length) {
          local_best_length[tid].length = length;
          memcpy(&local_best_path[(size_t)tid * (size_t)N_POINTS],
                 path, sizeof(int) * (size_t)N_POINTS);
        }

        // Update thread-local pheromone
        for (int i = 0; i < N_POINTS - 1; i++) {
          size_t a = (size_t)path[i];
          size_t b = (size_t)path[i + 1];
          dpriv[IDX(a, b, N_POINTS)] += Q / length;
        }
        // Complete tour (last -> first)
        dpriv[IDX((size_t)path[N_POINTS - 1], (size_t)path[0], N_POINTS)] += Q / length;
      }
    } // end parallel

    // Merge pheromone updates
    memset(delta_pheromone, 0, NN * sizeof(double));
    for (int t = 0; t < n_threads; t++) {
      double *src = delta_pheromone_private + (size_t)t * NN;
      for (size_t k = 0; k < NN; ++k) delta_pheromone[k] += src[k];
    }

    // Apply evaporation and deposit
    for (size_t k = 0; k < NN; ++k) {
      pheromone[k] *= EVAPORATION_RATE;
      pheromone[k] += delta_pheromone[k];
    }

    // Merge local bests into global best
    for (int t = 0; t < n_threads; t++) {
      if (local_best_length[t].length < global_best_length) {
        global_best_length = local_best_length[t].length;
        memcpy(global_best_path,
               &local_best_path[(size_t)t * (size_t)N_POINTS],
               sizeof(int) * (size_t)N_POINTS);
      }
    }
  }

  // Clean up heap (except the returned path)
  free(pheromone);
  free(delta_pheromone);
  free(delta_pheromone_private);
  free(local_best_length);
  free(local_best_path);

  ACOResult result = { global_best_length, global_best_path };
  return result;
}