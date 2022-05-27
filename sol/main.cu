#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "helper.h"

using namespace std;

#define PREALLOC_SIZE 1000
#define BLOCK_SIZE_GPU 256

/**
 * @brief Kernel function where each GPU thread gets assigned a city and finds 
 * all of the other cities in the range specified, hence adding their 
 * populations.
 *
 * @param total_pops array of cities' total population count
 * @param lats array of cities' latitudes
 * @param lons array of cities' longitudes
 * @param pops array of cities' population count
 * @param kmRange radius of search
 * @param n number of cities
 * @return __global__ none
 */
__global__ void add_populations(int *total_pops, const float *lats,
                                const float *lons, const int *pops,
                                const int kmRange, const size_t n) {
    // Get current city assigned to the GPU thread
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int *src_total_pop = &total_pops[idx];
    const float *src_lat = &lats[idx];
    const float *src_lon = &lons[idx];
    const int *src_pop = &pops[idx];

    // Check for outer bound
    if (idx >= n) return;

    /*
    * Finds the cities in range of the thread's city and adds their population.
    * The city that is found will have the thread's population added to it, 
    * thus the check is done only with further cities in the vector.
    */
    for (unsigned int i = idx + 1; i < n; ++i) {
        if (geoDistance(*src_lat, *src_lon, lats[i], lons[i]) <= kmRange) {
            atomicAdd(src_total_pop, pops[i]);
            atomicAdd(&total_pops[i], *src_pop);
        }
    }
}

/**
 * @brief CPU function that processes cities' data and outputs total population
 * count of each city's radius.
 *
 * @param kmRange city radius
 * @param fileIn input file name to read cities' data
 * @param fileOut output file name to write cities' total population count
 */
void process_cities(float kmRange, const char *fileIn, const char *fileOut) {
    string geon;
    float lat, lon;
    int pop;
    float *latsHost, *latsDevice, *lonsHost, *lonsDevice;
    int *popsHost, *popsDevice, *total_popsHost, *total_popsDevice;
    unsigned int num_cities = 0, capacity = PREALLOC_SIZE;
    vector<string> geons;

    // Open IO files
    ifstream ifs(fileIn);
    ofstream ofs(fileOut);

    // Initialize host heap memory
    latsHost = (float *)malloc(capacity * sizeof(float));
    lonsHost = (float *)malloc(capacity * sizeof(float));
    popsHost = (int *)malloc(capacity * sizeof(int));
    total_popsHost = (int *)malloc(capacity * sizeof(int));

    if (total_popsHost == NULL || popsHost == NULL || lonsHost == NULL ||
        latsHost == NULL) {
        cout << "Host memory allocation error.\n";
        return;
    }

    // Read data from input file
    while (ifs >> geon >> lat >> lon >> pop) {
        // Check if more memory needs to be reserved
        if (num_cities >= capacity) {
            capacity *= 2;
            latsHost = (float *)realloc(latsHost, sizeof(float) * capacity);
            lonsHost = (float *)realloc(lonsHost, sizeof(float) * capacity);
            popsHost = (int *)realloc(popsHost, sizeof(int) * capacity);
            total_popsHost =
                (int *)realloc(total_popsHost, sizeof(int) * capacity);

            if (total_popsHost == NULL || popsHost == NULL ||
                lonsHost == NULL || latsHost == NULL) {
                cout << "Host memory reallocation error.\n";
                return;
            }
        }

        // Add data to vectors
        geons.push_back(geon);
        latsHost[num_cities] = lat;
        lonsHost[num_cities] = lon;
        popsHost[num_cities] = pop;

        // Add city's population to total population count
        total_popsHost[num_cities] = pop;

        ++num_cities;
    }

    // Initialize device heap memory
    cudaMalloc((void **)&latsDevice, num_cities * sizeof(float));
    cudaMalloc((void **)&lonsDevice, num_cities * sizeof(float));
    cudaMalloc((void **)&popsDevice, num_cities * sizeof(int));
    cudaMalloc((void **)&total_popsDevice, num_cities * sizeof(int));

    if (total_popsDevice == NULL || popsDevice == NULL || lonsDevice == NULL ||
        latsDevice == NULL) {
        cout << "Device memory allocation error.\n";
        return;
    }

    // Copy data from Host to Device
    cudaMemcpy(latsDevice, latsHost, num_cities * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lonsDevice, lonsHost, num_cities * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(popsDevice, popsHost, num_cities * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(total_popsDevice, total_popsHost, num_cities * sizeof(int),
               cudaMemcpyHostToDevice);

    // Choose block and grid size
    const size_t block_size = BLOCK_SIZE_GPU;
    size_t blocks_no = (num_cities + block_size - 1) / block_size;

    // Launch GPU Kernel to compute total population counts
    add_populations<<<blocks_no, block_size>>>(total_popsDevice, latsDevice,
                                               lonsDevice, popsDevice, kmRange,
                                               num_cities);
    if (cudaSuccess != cudaDeviceSynchronize()) {
        cout << "Cuda Synchronize\n";
        return;
    }

    // Copy results from the Device to the Host
    cudaMemcpy(total_popsHost, total_popsDevice, num_cities * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Print results to output file
    for (unsigned int i = 0; i < num_cities; ++i) {
        ofs << total_popsHost[i] << '\n';
    }

    // Free Host memory
    free(latsHost);
    free(lonsHost);
    free(popsHost);
    free(total_popsHost);

    // Free Device Memory
    cudaFree(latsDevice);
    cudaFree(lonsDevice);
    cudaFree(popsDevice);
    cudaFree(total_popsDevice);

    // Close IO files
    ifs.close();
    ofs.close();
}

int main(int argc, char **argv) {
    DIE(argc == 1, "./accpop <kmrange1> <file1in> <file1out> ...");
    DIE((argc - 1) % 3 != 0, "./accpop <kmrange1> <file1in> <file1out> ...");

    for (int argcID = 1; argcID < argc - 3; argcID += 3) {
        float kmRange = atof(argv[argcID]);
        process_cities(kmRange, argv[argcID + 1], argv[argcID + 2]);
    }
}
