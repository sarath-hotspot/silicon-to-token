
#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1024;
    const size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    float h_a[1024], h_b[1024], h_c[1024];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vector_add<<<4, 256>>>(d_a, d_b, d_c, n);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Final output 
    std::cout << "c[11] = " << h_c[11] << '\n';

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
