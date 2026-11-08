#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                            \
            std::cerr << cudaGetErrorString(err) << "\n";     \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

int main() {
    constexpr int n = 1024;
    const size_t bytes = n * sizeof(float);
    std::vector<float> input(n);
    std::vector<float> output(n);

    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i);
    }

    float* device_data = nullptr;
    CUDA_CHECK(cudaMalloc(&device_data, bytes));
    CUDA_CHECK(cudaMemcpy(
        device_data, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        output.data(), device_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_data));

    if (output != input) {
        std::cerr << "Memory round-trip check failed\n";
        return 1;
    }

    std::cout << "Memory round-trip check passed\n";
    return 0;
}
