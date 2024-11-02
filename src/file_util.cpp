#include <cstdio>
#include <cstdlib>

#include "utils/model.cuh"
#include "utils/common.h"


#define MAX_LINE_LENGTH 1024

__host__ void readConvWeights(const char* filename) {
    FILE* weightsFile = fopen(filename, "r");
    model::NeckLayer neck;
    if (!weightsFile) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int current_layer = -1;
    int reading_weights = 0;
    int reading_biases = 0;
    int row_count = 0;

    while (fgets(line, MAX_LINE_LENGTH, weightsFile)) {
        if (strncmp(line, "Layer ", 6) == 0) {
            current_layer = line[6] - '0';  
            row_count = 0;
            continue;
        }

        if (strcmp(line, "Weights:") == 0) {
            reading_weights = 1;
            reading_biases = 0;
            row_count = 0;
            continue;
        }

        if (strcmp(line, "Biases:") == 0) {
            reading_weights = 0;
            reading_biases = 1;
            row_count = 0;
            continue;
        }

        if (strncmp(line, "Shape:", 6) == 0 || line[0] == '=') {
            continue;
        }

        if (reading_weights && strlen(line) > 0) {
            char* token = strtok(line, " ");
            int col = 0;
            
            switch (current_layer) {
                case 0:
                    while (token && col < model::Nout) {
                        neck.conv1[row_count][col][0][0] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    if (row_count < model::Nin1 - 1) row_count++;
                    break;
                case 1:
                    while (token && col < model::Nout) {
                        neck.conv2[row_count][col][0][0] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    if (row_count < model::Nin2 - 1) row_count++;
                    break;
                case 2:
                    while (token && col < model::Nout) {
                        neck.conv3[row_count][col][0][0] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    if (row_count < model::Nin3 - 1) row_count++;
                    break;
                case 3:
                    while (token && col < model::Nout) {
                        neck.conv4[row_count][col][0][0] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    if (row_count < model::Nin4 - 1) row_count++;
                    break;
            }
        }


        if (reading_biases && strlen(line) > 0) {
            char* token = strtok(line, " ");
            int col = 0;
            
            switch (current_layer) {
                case 0:
                    while (token && col < model::Nout) {
                        neck.bias1[col] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    break;
                case 1:
                    while (token && col < model::Nout) {
                        neck.bias2[col] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    break;
                case 2:
                    while (token && col < model::Nout) {
                        neck.bias3[col] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    break;
                case 3:
                    while (token && col < model::Nout) {
                        neck.bias4[col] = (floatT)atof(token);
                        token = strtok(NULL, " ");
                        col++;
                    }
                    break;
            }
        }
    }

    printf("Done reading weights and biases\n");
    printf("conv1: %f\n", static_cast<float>(neck.conv1[0][10][0][0]));
    fclose(weightsFile);
}

