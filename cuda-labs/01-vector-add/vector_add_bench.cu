
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                              \
            std::cerr << cudaGetErrorString(err) << "\n";      \
            std::exit(EXIT_FAILURE);                           \
        }                                                     \
    } while (0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_grid_stride(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    int n,
    int blocks,
    int threads,
    bool grid_stride
) {
    if (grid_stride) {
        vector_add_grid_stride<<<blocks, threads>>>(a, b, c, n);
    } else {
        vector_add<<<blocks, threads>>>(a, b, c, n);
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : (1 << 26);
    int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    std::string mode = argc > 3 ? argv[3] : "direct";

    if (n <= 0 || threads <= 0 || threads > 1024 ||
        threads % 32 != 0 ||
        (mode != "direct" && mode != "grid_stride")) {
        std::cerr << "Usage: " << argv[0]
                  << " [n] [threads] [direct|grid_stride]\n";
        return 1;
    }

    bool grid_stride = mode == "grid_stride";

    int device = 0;
    cudaDeviceProp properties;
    int runtime_version = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.125f;
        h_b[i] = static_cast<float>(i % 127) * 0.25f;
    }

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (n - 1) / threads + 1;
    if (grid_stride) {
        // A smaller grid makes each thread process multiple elements.
        blocks = std::min(blocks, properties.multiProcessorCount * 4);
    }

    // Warm up the GPU and kernel.
    for (int i = 0; i < 10; ++i) {
        launch_vector_add(
            d_a, d_b, d_c, n, blocks, threads, grid_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int iterations = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launch_vector_add(
            d_a, d_b, d_c, n, blocks, threads, grid_stride);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(
        h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        float cpu_result = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - cpu_result) > 1e-5f) {
            std::cerr << "Correctness check failed\n";
            return 1;
        }
    }

    double avg_ms = elapsed_ms / iterations;

    // Two reads and one write per element.
    double logical_bytes = 3.0 * bytes;

    double bandwidth_gbps =
        logical_bytes / (avg_ms / 1000.0) / 1e9;

    std::cout << "Kernel=" << mode << "\n"
              << "gpu=" << properties.name << "\n"
              << "cuda_runtime=" << runtime_version / 1000 << "."
              << (runtime_version % 1000) / 10 << "\n"
              << "cuda_driver=" << driver_version / 1000 << "."
              << (driver_version % 1000) / 10 << "\n"
              << "n=" << n << "\n"
              << "threads=" << threads << "\n"
              << "blocks=" << blocks << "\n"
              << "latency_ms=" << avg_ms << "\n"
              << "bandwidth_GBps=" << bandwidth_gbps << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
