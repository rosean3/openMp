#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

// Distance between two 3D points
double distance(int i, int j, double points[][3]);

// Roulette wheel selection
int roulette_wheel(int n, double *prob);

// Visualize best path with gnuplot
void visualize_path(int N_POINTS, double points[][3], int *best_path);

int read_points(const char *filename, double points[][3], int max_points);

#endif
