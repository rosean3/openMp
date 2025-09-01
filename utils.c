#include "utils.h"
#include <math.h>
#include <time.h>

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