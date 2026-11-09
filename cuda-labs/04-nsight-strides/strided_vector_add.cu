#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string_view>

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                   \
            std::cerr << #call << ": " << cudaGetErrorString(err)     \
                      << '\n';                                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                            \
    } while (0)

constexpr int threads = 256;
constexpr int warmups = 10;
constexpr int iterations = 100;

float input_a(size_t i) {
    return static_cast<float>(static_cast<int>(i % 251) - 125) * 0.125f;
}

float input_b(size_t i) {
    return static_cast<float>(i % 127) * 0.25f;
}

__global__ void strided_vector_add(
    const float* a, const float* b, float* c, size_t n, size_t stride
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t source = i * stride;
    if (i < n && source < n) {
        c[i] = a[source] + b[source];
    }
}

bool parse_positive(std::string_view text, size_t& value) {
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size()
        && value > 0;
}

int main(int argc, char** argv) {
    size_t n = 1000000000;
    size_t stride = 1;
    if (argc > 3 || (argc > 1 && !parse_positive(argv[1], n))
        || (argc > 2 && !parse_positive(argv[2], stride))
        || stride > 5 || n > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::cerr << "Usage: " << argv[0] << " [n > 0] [stride: 1..5]\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t bytes = n * sizeof(float);
    constexpr size_t output_bytes = threads * sizeof(float);
    if (free_bytes < output_bytes || bytes > (free_bytes - output_bytes) / 2) {
        std::cerr << "Insufficient free GPU memory for two " << bytes
                  << "-byte inputs and a " << output_bytes << "-byte output; "
                  << free_bytes << " bytes free\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr<float[]> host_a(new (std::nothrow) float[n]);
    std::unique_ptr<float[]> host_b(new (std::nothrow) float[n]);
    if (!host_a || !host_b) {
        std::cerr << "Failed to allocate two " << bytes << "-byte host inputs\n";
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < n; ++i) {
        host_a[i] = input_a(i);
        host_b[i] = input_b(i);
    }

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, output_bytes));
    CUDA_CHECK(cudaMemcpy(a, host_a.get(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host_b.get(), bytes, cudaMemcpyHostToDevice));
    std::array<float, threads> output;
    output.fill(-1.0f);
    CUDA_CHECK(cudaMemcpy(c, output.data(), output_bytes, cudaMemcpyHostToDevice));

    for (int repeat = 0; repeat < warmups; ++repeat) {
        strided_vector_add<<<1, threads>>>(a, b, c, n, stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < iterations; ++repeat) {
        strided_vector_add<<<1, threads>>>(a, b, c, n, stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(output.data(), c, output_bytes, cudaMemcpyDeviceToHost));

    size_t active_threads = std::min(static_cast<size_t>(threads), (n - 1) / stride + 1);
    for (size_t i = 0; i < output.size(); ++i) {
        float expected = i < active_threads
            ? input_a(i * stride) + input_b(i * stride) : -1.0f;
        if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > 1e-5f) {
            std::cerr << "Correctness check failed at i=" << i
                      << ": expected " << expected << ", got " << output[i] << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "gpu=" << properties.name << '\n'
              << "n=" << n << '\n'
              << "stride=" << stride << '\n'
              << "blocks=1\nthreads=" << threads << '\n'
              << "active_threads=" << active_threads << '\n'
              << "warmups=" << warmups << "\niterations=" << iterations << '\n'
              << "latency_ms=" << elapsed_ms / iterations << '\n'
              << "All output and bounds checks passed\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return EXIT_SUCCESS;
}
