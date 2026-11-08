#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                            \
            std::cerr << cudaGetErrorString(err) << "\n";     \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

using Clock = std::chrono::steady_clock;
constexpr int warmups = 5;
constexpr int iterations = 20;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void copyAndWait(void* destination, const void* source, size_t bytes,
                 cudaMemcpyKind direction) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, direction));
    // Pageable H2D copies can return before the device transfer completes.
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void checkResult(const float* input, const float* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (input[i] != output[i]) {
            std::cerr << "Transfer correctness check failed at index " << i
                      << ": " << input[i] << " != " << output[i] << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

double timeCopies(void* destination, const void* source, size_t bytes,
                  cudaMemcpyKind direction) {
    for (int i = 0; i < warmups; ++i) {
        copyAndWait(destination, source, bytes, direction);
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        copyAndWait(destination, source, bytes, direction);
    }
    return elapsedMs(start, Clock::now()) / iterations;
}

void benchmark(float* deviceData, size_t n, bool pinned, bool cold) {
    size_t bytes = n * sizeof(float);
    float* input = nullptr;
    float* output = nullptr;

    auto allocationStart = Clock::now();
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&input, bytes));
        CUDA_CHECK(cudaMallocHost(&output, bytes));
    } else {
        input = static_cast<float*>(std::malloc(bytes));
        output = static_cast<float*>(std::malloc(bytes));
        if (input == nullptr || output == nullptr) {
            std::free(input);
            std::free(output);
            std::cerr << "Failed to allocate pageable host buffers\n";
            std::exit(EXIT_FAILURE);
        }
    }
    double allocationMs = elapsedMs(allocationStart, Clock::now());

    auto initializationStart = Clock::now();
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 251) * 0.25f;
        output[i] = -1.0f;
    }
    double initializationMs = elapsedMs(initializationStart, Clock::now());

    auto transferStart = Clock::now();
    copyAndWait(deviceData, input, bytes, cudaMemcpyHostToDevice);
    auto h2dStop = Clock::now();
    copyAndWait(output, deviceData, bytes, cudaMemcpyDeviceToHost);
    auto roundTripStop = Clock::now();
    double coldElapsedMs = elapsedMs(allocationStart, roundTripStop);
    checkResult(input, output, n);

    double h2dMs = elapsedMs(transferStart, h2dStop);
    double d2hMs = elapsedMs(h2dStop, roundTripStop);
    if (!cold) {
        h2dMs = timeCopies(deviceData, input, bytes, cudaMemcpyHostToDevice);
        d2hMs = timeCopies(output, deviceData, bytes, cudaMemcpyDeviceToHost);
        checkResult(input, output, n);
    }

    if (pinned) {
        CUDA_CHECK(cudaFreeHost(input));
        CUDA_CHECK(cudaFreeHost(output));
    } else {
        std::free(input);
        std::free(output);
    }

    std::cout << "Memory=" << (pinned ? "pinned" : "pageable") << "\n"
              << "host_allocation_ms=" << allocationMs << "\n"
              << "host_initialization_ms=" << initializationMs << "\n"
              << "H2D_elapsed_ms=" << h2dMs << "\n"
              << "H2D_bandwidth_GBps=" << bytes / (h2dMs * 1e6) << "\n"
              << "D2H_elapsed_ms=" << d2hMs << "\n"
              << "D2H_bandwidth_GBps=" << bytes / (d2hMs * 1e6) << "\n";
    if (cold) {
        std::cout << "cold_setup_round_trip_ms=" << coldElapsedMs << "\n";
    }
}

int main(int argc, char** argv) {
    size_t n = 1 << 24;
    bool valid = argc <= 4;
    if (argc >= 2) {
        std::string_view argument(argv[1]);
        auto result = std::from_chars(
            argument.data(), argument.data() + argument.size(), n);
        valid = result.ec == std::errc{} &&
                result.ptr == argument.data() + argument.size();
    }
    std::string_view firstMemory = argc >= 3 ? argv[2] : "pageable";
    std::string_view experiment = argc == 4 ? argv[3] : "warm";
    if (!valid || n == 0 ||
        n > std::numeric_limits<size_t>::max() / sizeof(float) ||
        (firstMemory != "pageable" && firstMemory != "pinned") ||
        (experiment != "warm" && experiment != "cold")) {
        std::cerr << "Usage: " << argv[0]
                  << " [positive FP32 element count] [pageable|pinned first]"
                  << " [warm|cold]\n";
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    size_t bytes = n * sizeof(float);
    float* deviceData = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceData, bytes));

    bool cold = experiment == "cold";
    std::cout << "gpu=" << properties.name << "\n"
              << "n=" << n << "\n"
              << "bytes_per_transfer=" << bytes << "\n"
              << "warmups=" << (cold ? 0 : warmups) << "\n"
              << "iterations=" << (cold ? 1 : iterations) << "\n"
              << "experiment=" << experiment << "\n";

    bool pinnedFirst = firstMemory == "pinned";
    benchmark(deviceData, n, pinnedFirst, cold);
    benchmark(deviceData, n, !pinnedFirst, cold);
    CUDA_CHECK(cudaFree(deviceData));
    std::cout << "All transfer correctness checks passed\n";
    return 0;
}
