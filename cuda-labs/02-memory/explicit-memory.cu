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

//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    if(comparisonResult == nullptr)
    {
        fprintf(stderr, "Failed to allocate CPU comparison buffer\n");
        std::exit(EXIT_FAILURE);
    }
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    CUDA_CHECK(cudaMallocHost(&A, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C, vectorLength*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devC, 0, vectorLength*sizeof(float)));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    cudaEvent_t gpuStart, gpuStop;
    CUDA_CHECK(cudaEventCreate(&gpuStart));
    CUDA_CHECK(cudaEventCreate(&gpuStop));
    CUDA_CHECK(cudaEventRecord(gpuStart));
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(gpuStop));
    // wait for kernel execution to complete
    CUDA_CHECK(cudaEventSynchronize(gpuStop));
    float gpuElapsedMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuElapsedMs, gpuStart, gpuStop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault));

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
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    CUDA_CHECK(cudaEventDestroy(gpuStart));
    CUDA_CHECK(cudaEventDestroy(gpuStop));
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C));
    free(comparisonResult);

    if(!resultsMatch)
    {
        std::exit(EXIT_FAILURE);
    }

    printf("GPU elapsed time (ms)=%.6f\n", gpuElapsedMs);
    printf("CPU elapsed time (ms)=%.6f\n", cpuElapsedMs);
}
//explicit-memory-end


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
    explicitMemExample(vectorLength);		
    return 0;
}