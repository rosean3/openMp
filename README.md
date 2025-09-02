# Projects
- Dot Product
- Travelling Sales Person (TSP) problem solved with Ant Colony Optimization (ACO)

## Dot Product with OpenMP

This project demonstrates the computation of the **dot product** between two large vectors, with the option to run in **parallel** (using OpenMP) or **sequentially**. It provides a simple comparison of performance and correctness between the two execution modes.

---

### 📋 Requirements
- GCC with OpenMP support  
- A system with multiple CPU cores (to benefit from parallel execution)  

---

### ⚙️ Compilation

Compile the program with:

```bash
gcc dot_product.c -o dot_product -fopenmp
```

### Run the program:
```bash
./dot_product
```

You will be prompted to choose the execution mode:
`Do you wish to run the parallel or sequential version (p/s)?:`
- Enter `p` for parallel execution.

- Enter `s` for sequential execution.

### 📊 Example Results
#### Parallel execution
```bash
./dot_product
Do you wish to run the parallel or sequential version (p/s)?: p
The dot product is: 14814785180646746112.000000
Elapsed time: 1.621 seconds
```

#### Sequential execution
```bash
./dot_product
Do you wish to run the parallel or sequential version (p/s)?: s
The dot product is: 14814785174975741952.000000
Elapsed time: 1.723 seconds
```

### 📈 Observations

The dot product results are nearly identical (differences are due to floating-point rounding).

The parallel version executes slightly faster than the sequential one, demonstrating the benefits of OpenMP parallelization on large data sizes.

With larger arrays or more CPU cores, the performance difference can become more significant.

## Travelling Sales Person (TSP) with Ant Colony Optimization (ACO)
This project implements the Travelling Sales Person problem using Ant Colony Optimization (ACO). The program can compute solutions in parallel (using OpenMP) or sequentially, and optionally visualize the best path found.

### 📋 Requirements

- GCC with OpenMP support

- gnuplot for visualization (optional)

- A system with multiple CPU cores for parallel execution

### ⚙️ Compilation
```bash
gcc aco.c utils.c -o aco -lm -fopenmp
```

### Run the program
```bash
./aco
```

You will be prompted to provide:

- Number of points (cities).

- Number of iterations for the ACO algorithm.

- Whether to visualize the result (0 = no, 1 = yes).

- Execution mode: parallel or sequential (p/s).

### 📊 Example Results
#### Parallel execution

#### Sequential execution
