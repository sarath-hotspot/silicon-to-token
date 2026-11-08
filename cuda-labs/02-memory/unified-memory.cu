#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                            \
            fprintf(stderr, "%s\n", cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    if(comparisonResult == nullptr)
    {
        fprintf(stderr, "Failed to allocate CPU comparison buffer\n");
        std::exit(EXIT_FAILURE);
    }

    // Use unified memory to allocate buffers
    CUDA_CHECK(cudaMallocManaged(&A, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, vectorLength*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    cudaEvent_t gpuStart, gpuStop;
    CUDA_CHECK(cudaEventCreate(&gpuStart));
    CUDA_CHECK(cudaEventCreate(&gpuStop));
    CUDA_CHECK(cudaEventRecord(gpuStart));
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(gpuStop));
    // Wait for the kernel to complete execution
    CUDA_CHECK(cudaEventSynchronize(gpuStop));
    float gpuElapsedMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuElapsedMs, gpuStart, gpuStop));

    // Perform computation serially on CPU for comparison
    auto cpuStart = std::chrono::steady_clock::now();
    serialVecAdd(A, B, comparisonResult, vectorLength);
    auto cpuStop = std::chrono::steady_clock::now();
    double cpuElapsedMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    // Confirm that CPU and GPU got the same answer
    bool resultsMatch =
        vectorApproximatelyEqual(C, comparisonResult, vectorLength);
    if(resultsMatch)
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    CUDA_CHECK(cudaEventDestroy(gpuStart));
    CUDA_CHECK(cudaEventDestroy(gpuStop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    free(comparisonResult);

    if(!resultsMatch)
    {
        std::exit(EXIT_FAILURE);
    }

    printf("GPU elapsed time (ms)=%.6f\n", gpuElapsedMs);
    printf("CPU elapsed time (ms)=%.6f\n", cpuElapsedMs);
}
//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    if(vectorLength <= 0)
    {
        fprintf(stderr, "Usage: %s [positive vector length]\n", argv[0]);
        return 1;
    }
    unifiedMemExample(vectorLength);		
    return 0;
}