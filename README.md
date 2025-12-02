# Trabalho Computação Paralela
- Felipe
- Luis
- Pedro

## Algorítimo Sequencial

```bash
gcc raw_kohonen_som_trace.c -o raw_kohonen_som_trace
time ./raw_kohonen_som_trace
mnist_test.csv
```
```text
Enter CSV file path: mnist_test.csv
Warning: Line 1 has 19 columns, expected 785
Info: Reached maximum line limit of 1024 lines
Successfully read 1024 rows from 'mnist_test.csv'

Sample data (first 3 rows):
Row 1: Label = 10, First 3 values = [28, 28, 28, ...]
Row 2: Label = 7, First 3 values = [0, 0, 0, ...]
Row 3: Label = 2, First 3 values = [0, 0, 0, ...]

Data range: 0 to 255

Running Kohonen SOM training...
Parameters: 1024 samples, 784 features, 100 output neurons
Successfully saved trained weights to 'som_weights_output.bin'

Kohonen SOM processing completed.

real    0m58.958s
user    0m33.115s
sys     0m0.005s
```

## Algorítimo com OpenMP p/ CPU

```bash
gcc openmp_cpu_kohonen_som_trace.c -o openmp_cpu_kohonen_som_trace -fopenmp
time ./openmp_cpu_kohonen_som_trace
mnist_test.csv
```
```text
Enter CSV file path: mnist_test.csv
Warning: Line 1 has 19 columns, expected 785
Info: Reached maximum line limit of 1024 lines
Successfully read 1024 rows from 'mnist_test.csv'

Sample data (first 3 rows):
Row 1: Label = 10, First 3 values = [28, 28, 28, ...]
Row 2: Label = 7, First 3 values = [0, 0, 0, ...]
Row 3: Label = 2, First 3 values = [0, 0, 0, ...]

Data range: 0 to 255

Running Kohonen SOM training...
Parameters: 1024 samples, 784 features, 100 output neurons
Successfully saved trained weights to 'som_weights_output.bin'

Kohonen SOM processing completed.

real    0m41.470s
user    0m33.902s
sys     0m0.009s
```
- **Speedup (p/ Sequencial)**: 42%

## Algorítimo com OpenMP p/ GPU

```bash
gcc -O3 openmp_gpu_kohonen_som_trace.c -o openmp_gpu_kohonen_som_trace -fopenmp
time ./openmp_gpu_kohonen_som_trace
mnist_test.csv
```
```text
Enter CSV file path: mnist_test.csv
Warning: Line 1 has 19 columns, expected 785
Info: Reached maximum line limit of 1024 lines
Successfully read 1024 rows from 'mnist_test.csv'

Sample data (first 3 rows):
Row 1: Label = 10, First 3 values = [28, 28, 28, ...]
Row 2: Label = 7, First 3 values = [0, 0, 0, ...]
Row 3: Label = 2, First 3 values = [0, 0, 0, ...]

Data range: 0 to 255

Running Kohonen SOM training...
Parameters: 1024 samples, 784 features, 100 output neurons
Successfully saved trained weights to 'som_weights_output.bin'

Kohonen SOM processing completed.

real    0m10.501s
user    0m0.086s
sys     0m0.035s
```
- **Speedup (p/ Sequencial)**: 461%

## Algorítimo com CUDA