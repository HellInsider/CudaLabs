
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixMultiplication.h"

#include <cmath>
#include <iostream>

inline __device__ unsigned get_lane_id() {
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

inline __device__ unsigned get_warp_id() {
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__global__ void simpleMultKernel(float* c, const float* a, const float* b, unsigned int L, unsigned int M, unsigned int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < L && col < N)
    {
        float sum = 0;
        for (int i = 0; i < M; i++) {
            sum += a[row * M + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

static constexpr unsigned s_tileWidth = 32;

__global__ void sharedMemoryMultKernel(float* c, const float* a, const float* b, unsigned int L, unsigned int M, unsigned int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sharedA[s_tileWidth][s_tileWidth];
    __shared__ float sharedB[s_tileWidth][s_tileWidth];

    if (row < L && col < N)
    {
        float cVal = 0;
        for (int i = 0; i < M; i += blockDim.x)
        { 
            sharedA[ty][tx] = a[row * M + i + tx];
            sharedB[ty][tx] = b[i * N + ty * N + col];
            __syncthreads();
            for (unsigned int j = 0; j < s_tileWidth; j++)
                cVal += sharedA[ty][j] * sharedB[j][tx];
            __syncthreads();
        }
        c[row * N + col] = cVal;
    }
}

__global__ void warpMultKernel(float* c, const float* a, const float* b, unsigned int L, unsigned int M, unsigned int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < L && col < N)
    {
        float sum = 0;
        for (int i = 0; i < M; i += warpSize) {
            unsigned laneId = get_lane_id();
            float aVal = a[row * M + i + laneId];
            float bVal = b[(i + laneId) * N + col];

            for (int j = 0; j < warpSize; j++) {
                sum += __shfl_sync(0xffffffff, a[row * M + i + j], 0) * __shfl_sync(0xffffffff, b[(i + j) * N + col], 0);
            }
        }
        c[row * N + col] = sum;
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t matrixMult(float* c, const float* a, const float* b, unsigned int L, unsigned int M, unsigned int N, MatrixMultiplicationMode mode)
{
    float* dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, L * N  * sizeof(decltype(dev_c[0])));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, L * M * sizeof(decltype(dev_a[0])));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, M * N * sizeof(decltype(dev_b[0])));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, L * M * sizeof(decltype(dev_a[0])), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, M * N * sizeof(decltype(dev_b[0])), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(N, L);
    dim3 blocksPerGrid(1, 1);
    if (L * N > 512)
    {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = static_cast<unsigned>(std::ceil(double(N) / double(threadsPerBlock.x)));
        blocksPerGrid.y = static_cast<unsigned>(std::ceil(double(L) / double(threadsPerBlock.y)));
    }

    switch (mode)
    {
    case MatrixMultiplicationMode::Simple:
        simpleMultKernel <<< threadsPerBlock, blocksPerGrid >>> (dev_c, dev_a, dev_b, L, M, N);
        break;
    case MatrixMultiplicationMode::SharedMemory:
        sharedMemoryMultKernel <<< threadsPerBlock, blocksPerGrid >>> (dev_c, dev_a, dev_b, L, M, N);
        break;
    case MatrixMultiplicationMode::WarpInstrincts:
        warpMultKernel <<< threadsPerBlock, blocksPerGrid >>> (dev_c, dev_a, dev_b, L, M, N);
        break;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, L * N * sizeof(decltype(dev_c[0])), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
