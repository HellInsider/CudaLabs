#include "MatrixMultiplication.h"
#include <memory>

#include <stdio.h>
#include <iostream>

void matMultiplyOnHost(float* A, float* B, float* C, int L, int M, int N) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < M; k++) {
                C[i * N + j] += A[i * M + k] * B[k * N + j];
            }
        }
    }
    return;
}

void printMatrix(float* a, int l, int m) {
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << a[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
    return;
}

void compareMatrix(float* origin, float* b, int l, int m) {
    for (int i = 0; i < l * m; i++) {
        if (origin[i] != b[i]) {
            printf("Simple Mismatch at Row = %d Col = %d origin = %f --device[] %f. \n Error.\n", i / m,
                i % m, origin[i], b[i]);
            break;
        }
    }
    std::cout << "Everything nice!" << std::endl;
}

int main()
{
    constexpr unsigned int L = 3, M = 3, N = 3;
    float* a;
    float* b;
    float* cCheck;
    float* cSimple;
    float* cShared;
    float* cWarp;

    a = (float*)malloc(sizeof(float) * L * M);
    b = (float*)malloc(sizeof(float) * M * N);

    cCheck = (float*)malloc(sizeof(float) * L * N);
    cSimple = (float*)malloc(sizeof(float) * L * N);
    cShared = (float*)malloc(sizeof(float) * L * N);
    cWarp = (float*)malloc(sizeof(float) * L * N);
   
    for (int i = 0; i < L * M; i++) {
        a[i] = (float)(rand() % 99);
    }
    std::cout << "a = {" << std::endl;
    printMatrix(a, L, M);
    std::cout << "}" << std::endl;

    for (int i = 0; i < M * N; i++) {
        b[i] = (float)(rand() % 99);
    }
    std::cout << "b = {"<< std::endl;
    printMatrix(b, M, N);
    std::cout << "}" << std::endl;

    matMultiplyOnHost(a, b, cCheck, L, M, N);

    std::cout << "Correct mul res = {" << std::endl;
    printMatrix(cCheck, L, N);
    std::cout << "}" << std::endl;

    cudaError_t cudaStatus = matrixMult(cSimple, a, b, L, M, N, MatrixMultiplicationMode::Simple);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "process failed!");
        return 1;
    }
    
    compareMatrix(cCheck, cSimple, L, N);
    std::cout << "Simple mul res = {" << std::endl;
    printMatrix(cSimple, L, N);
    std::cout << "}" << std::endl << std::endl;    

    cudaStatus = matrixMult(cShared, a, b, L, M, N, MatrixMultiplicationMode::SharedMemory);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "process failed!");
        return 1;
    }    
    compareMatrix(cCheck, cShared, L, N);
    std::cout << "Shared mul res = {" << std::endl;
    printMatrix(cShared, L, N);
    std::cout << "}" << std::endl;

    cudaStatus = matrixMult(cWarp, a, b, L, M, N, MatrixMultiplicationMode::WarpInstrincts);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "process failed!");
        return 1;
    }

    compareMatrix(cCheck, cWarp, L, N);
    std::cout << "Warp mul res = {" << std::endl;
    printMatrix(cWarp, L, N);
    std::cout << "}" << std::endl;

    free(a);
    free(b);
    free(cCheck);
    free(cSimple);
    free(cShared);
    free(cWarp);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}