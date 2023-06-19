#pragma once
#include "cuda_runtime.h"

enum class MatrixMultiplicationMode
{
	Simple,
	SharedMemory,
	WarpInstrincts
};

cudaError_t matrixMult(float* c, const float* a, const float* b, unsigned int L, unsigned int M, unsigned int N, MatrixMultiplicationMode mode);
