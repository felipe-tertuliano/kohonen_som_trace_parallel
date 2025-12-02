#define _USE_MATH_DEFINES
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef max
/** shorthand for maximum value */
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
/** shorthand for minimum value */
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define NUM_COLUMNS 785 // 1 label + 784 values
#define MAX_LINE_LENGTH 4096 // Large enough to hold all columns
#define MAX_LINES 1024 // Maximum number of data lines to process

/**
 * Reads CSV data from a file and extracts labels and pixel values
 * \param[in] filename Path to the CSV file to read
 * \param[out] labels Pointer to array where labels will be stored (1D array)
 * \param[out] data Pointer to 2D array where pixel values will be stored
 * \param[out] num_rows Number of data rows (excluding header) read from file
 * \return 1 on success, 0 on failure
 */
int read_csv_data(const char *filename, double **labels, double ***data, int *num_rows)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file '%s'\n", filename);
        return 0;
    }

    char line[MAX_LINE_LENGTH];
    int capacity = (MAX_LINES < 256) ? MAX_LINES : 256;  // Initial capacity, capped by MAX_LINES
    int count = 0;
    
    // Allocate initial memory (limited by MAX_LINES)
    if (capacity > MAX_LINES) capacity = MAX_LINES;
    *labels = (double *)malloc(capacity * sizeof(double));
    *data = (double **)malloc(capacity * sizeof(double *));
    
    if (!*labels || !*data) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return 0;
    }
    
    // Read and skip the header line
    if (!fgets(line, sizeof(line), file)) {
        printf("Error: File is empty or header missing\n");
        fclose(file);
        return 0;
    }
    
    // Read each data line until MAX_LINES is reached
    while (fgets(line, sizeof(line), file) && count < MAX_LINES) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(line) == 0) continue;
        
        // Check if we need to resize arrays (but not beyond MAX_LINES)
        if (count >= capacity && count < MAX_LINES) {
            // Increase capacity, but cap at MAX_LINES
            int new_capacity = capacity * 2;
            if (new_capacity > MAX_LINES) new_capacity = MAX_LINES;
            
            // Only realloc if we're actually increasing capacity
            if (new_capacity > capacity) {
                *labels = (double *)realloc(*labels, new_capacity * sizeof(double));
                *data = (double **)realloc(*data, new_capacity * sizeof(double *));
            
                if (!*labels || !*data) {
                    printf("Error: Memory reallocation failed\n");
                    fclose(file);
                    return 0;
                }
                capacity = new_capacity;
            }
        }
        
        // Allocate memory for this row's data (784 values, excluding label)
        (*data)[count] = (double *)malloc(784 * sizeof(double));
        if (!(*data)[count]) {
            printf("Error: Memory allocation for row data failed\n");
            fclose(file);
            return 0;
        }
        
        // Parse the CSV line
        char *token;
        int col = 0;
        char *line_copy = strdup(line);  // Make a copy since strtok modifies the string
        char *line_ptr = line_copy;
        
        token = strtok(line_copy, ",");
        while (token != NULL && col < NUM_COLUMNS) {
            // Remove any whitespace
            while (isspace((unsigned char)*token)) token++;
            
            // Convert to double
            double value = atof(token);
            
            if (col == 0) {
                // First column is the label
                (*labels)[count] = value;
            } else {
                // Remaining columns are data values (0-255)
                (*data)[count][col - 1] = value;
            }
            
            token = strtok(NULL, ",");
            col++;
        }
        
        free(line_ptr);
        
        // Check if we read all columns
        if (col != NUM_COLUMNS) {
            printf("Warning: Line %d has %d columns, expected %d\n", count + 1, col, NUM_COLUMNS);
        }
        
        count++;
        
        // Check if we've reached MAX_LINES
        if (count >= MAX_LINES) {
            printf("Info: Reached maximum line limit of %d lines\n", MAX_LINES);
            break;
        }
    }
    
    *num_rows = count;
    fclose(file);
    
    // Trim arrays to exact size
    if (count < capacity) {
        *labels = (double *)realloc(*labels, count * sizeof(double));
        *data = (double **)realloc(*data, count * sizeof(double *));
    }
    
    return 1;
}

/**
 * Frees all memory allocated by read_csv_data()
 * \param[in] labels Pointer to labels array returned by read_csv_data()
 * \param[in] data Pointer to 2D data array returned by read_csv_data()
 * \param[in] num_rows Number of rows in the arrays
 */
void free_csv_data(double *labels, double **data, int num_rows)
{
    if (labels) free(labels);
    
    if (data) {
        for (int i = 0; i < num_rows; i++) {
            if (data[i]) free(data[i]);
        }
        free(data);
    }
}

/**
 * Save a given n-dimensional data martix to file
 * \param [in] fname filename to save in (gets overwriten without confirmation)
 * \param [in] X matrix to save
 * \param [in] num_points rows in the matrix = number of points
 * \param [in] num_features columns in the matrix = dimensions of points
 * \returns 0 if all ok
 * \returns -1 if file creation failed
 */
int save_nd_data(const char *fname, double **X, int num_points, int num_features)
{
    FILE *fp = fopen(fname, "wt");
    if (!fp)  // error with fopen
    {
        char msg[120];
        sprintf(msg, "File error (%s): ", fname);
        perror(msg);
        return -1;
    }

    for (int i = 0; i < num_points; i++)  // for each point in the array
    {
        for (int j = 0; j < num_features; j++)  // for each feature in the array
        {
            fprintf(fp, "%.4g", X[i][j]);  // print the feature value
            if (j < num_features - 1)      // if not the last feature
                fprintf(fp, ",");          // suffix comma
        }
        if (i < num_points - 1)  // if not the last row
            fprintf(fp, "\n");   // start a new line
    }
    fclose(fp);
    return 0;
}

/**
 * Get minimum value and index of the value in a vector
 * \param[in] X vector to search
 * \param[in] N number of points in the vector
 * \param[out] val minimum value found
 * \param[out] idx index where minimum value was found
 */
void kohonen_get_min_1d(double const *X, int N, double *val, int *idx)
{
    val[0] = INFINITY;  // initial min value

    for (int i = 0; i < N; i++)  // check each value
    {
        if (X[i] < val[0])  // if a lower value is found save the value and its index
        {
            idx[0] = i;
            val[0] = X[i];
        }
    }
}

/**
 * Update weights of the SOM using Kohonen algorithm
 * \param[in] x data point
 * \param[in,out] W weights matrix
 * \param[in,out] D temporary vector to store distances
 * \param[in] num_out number of output points
 * \param[in] num_features number of features per input sample
 * \param[in] alpha learning rate \f$0<\alpha\le1\f$
 * \param[in] R neighborhood range
 */
void kohonen_update_weights(double const *x, double *const *W, double *D, int num_out, int num_features, double alpha, int R)
{
    int j, k;

    // step 1: for each output point
    for (j = 0; j < num_out; j++)
    {
        D[j] = 0.f;
        // compute Euclidian distance of each output
        // point from the current sample
        for (k = 0; k < num_features; k++)
            D[j] += (W[j][k] - x[k]) * (W[j][k] - x[k]);
    }

    // step 2:  get closest node i.e., node with smallest Euclidian distance to
    // the current pattern
    int d_min_idx;
    double d_min;
    kohonen_get_min_1d(D, num_out, &d_min, &d_min_idx);

    // step 3a: get the neighborhood range
    int from_node = max(0, d_min_idx - R);
    int to_node = min(num_out, d_min_idx + R + 1);

    // step 3b: update the weights of nodes in the
    // neighborhood
    for (j = from_node; j < to_node; j++)
        for (k = 0; k < num_features; k++)
            // update weights of nodes in the neighborhood
            W[j][k] += alpha * (x[k] - W[j][k]);
}

/**
 * Apply incremental algorithm with updating neighborhood and learning rates on all samples in the given datset
 * \param[in] X data set
 * \param[in,out] W weights matrix
 * \param[in] num_samples number of output points
 * \param[in] num_features number of features per input sample
 * \param[in] num_out number of output points
 * \param[in] alpha_min terminal value of alpha
 */
void kohonen_som_tracer(double **X, double *const *W, int num_samples, int num_features, int num_out, double alpha_min)
{
    int R = num_out >> 2, iter = 0;
    double alpha = 1.f;
    double *D = (double *)malloc(num_out * sizeof(double));

    // Loop alpha from 1 to alpha_min
    for (; alpha > alpha_min; alpha -= 0.01, iter++)
    {
        // Loop for each sample pattern in the data set
        for (int sample = 0; sample < num_samples; sample++)
        {
            const double *x = X[sample];
            // update weights for the current input pattern sample
            kohonen_update_weights(x, W, D, num_out, num_features, alpha, R);
        }

        // every 10th iteration, reduce the neighborhood range
        if (iter % 10 == 0 && R > 1)
            R--;
    }

    free(D);
}

int main() {
    char filepath[256];
    double *labels = NULL;
    double **data = NULL;
    int num_rows = 0;
    
    // Get file path from user
    printf("Enter CSV file path: ");
    if (scanf("%255s", filepath) != 1) {
        printf("Error reading input\n");
        return 1;
    }
    
    // Read CSV data
    if (!read_csv_data(filepath, &labels, &data, &num_rows)) {
        return 1;
    }
    
    printf("Successfully read %d rows from '%s'\n", num_rows, filepath);
    
    // Display sample of the data (first 3 rows)
    printf("\nSample data (first 3 rows):\n");
    int display_rows = (num_rows < 3) ? num_rows : 3;
    
    for (int i = 0; i < display_rows; i++) {
        printf("Row %d: Label = %.0f, First 3 values = [%.0f, %.0f, %.0f, ...]\n", 
               i + 1, labels[i], 
               data[i][0], data[i][1], data[i][2]);
    }
    
    // Calculate some statistics
    double min_val = 255, max_val = 0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < 784; j++) {
            if (data[i][j] < min_val) min_val = data[i][j];
            if (data[i][j] > max_val) max_val = data[i][j];
        }
    }
    printf("\nData range: %.0f to %.0f\n", min_val, max_val);
    
    // Kohonen SOM parameters
    int num_features = 784;  // Assuming MNIST-like data (28x28 = 784 pixels)
    int num_out = 100;       // Number of output neurons (adjust as needed)
    double alpha_min = 0.01; // Terminal learning rate
    
    // Initialize weights matrix
    double **weights = (double **)malloc(num_out * sizeof(double *));
    if (weights == NULL) {
        printf("Error allocating weights matrix\n");
        free_csv_data(labels, data, num_rows);
        return 1;
    }
    
    for (int i = 0; i < num_out; i++) {
        weights[i] = (double *)malloc(num_features * sizeof(double));
        if (weights[i] == NULL) {
            printf("Error allocating weights row %d\n", i);
            // Free already allocated rows
            for (int j = 0; j < i; j++) {
                free(weights[j]);
            }
            free(weights);
            free_csv_data(labels, data, num_rows);
            return 1;
        }
        
        // Initialize weights randomly (between min_val and max_val)
        for (int j = 0; j < num_features; j++) {
            weights[i][j] = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
        }
    }
    
    printf("\nRunning Kohonen SOM training...\n");
    printf("Parameters: %d samples, %d features, %d output neurons\n", num_rows, num_features, num_out);
    
    // Apply Kohonen SOM algorithm
    kohonen_som_tracer(data, weights, num_rows, num_features, num_out, alpha_min);
    
    // Save the trained weights to a file
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "raw_kohonen_som_trace.bin");
    
    if (save_nd_data(output_file, weights, num_out, num_features) == 0) {
        printf("Successfully saved trained weights to '%s'\n", output_file);
    } else {
        printf("Failed to save weights to file\n");
    }
    
    // For demonstration, we can save the first 10 transformed points
    int sample_size = (num_rows < 10) ? num_rows : 10;
    double **sample_output = (double **)malloc(sample_size * sizeof(double *));
    
    if (sample_output != NULL) {
        for (int i = 0; i < sample_size; i++) {
            sample_output[i] = (double *)malloc(num_out * sizeof(double));
            if (sample_output[i] != NULL) {
                // Calculate the transformed representation for each sample
                // (This would need to be implemented based on how kohonen_som_tracer works)
                // For now, we'll just save a dummy version - you should replace this
                // with actual transformation logic
                for (int j = 0; j < num_out; j++) {
                    // This is placeholder - replace with actual transformation
                    sample_output[i][j] = 0.0;
                }
            }
        }
        
        // Save the sample (if you implement the transformation)
        // save_nd_data(sample_file, sample_output, sample_size, num_out);
        
        // Free sample output memory
        for (int i = 0; i < sample_size; i++) {
            free(sample_output[i]);
        }
        free(sample_output);
    }
    
    // Free allocated memory
    for (int i = 0; i < num_out; i++) {
        free(weights[i]);
    }
    free(weights);
    free_csv_data(labels, data, num_rows);
    
    printf("\nKohonen SOM processing completed.\n");
    
    return 0;
}
