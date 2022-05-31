#include "myRandom.hpp"
#include <iostream>

#define GPU 0
#define N   1000000
#define BLOCK_SIZE 1024


__global__ void kernel(double *,const uint*, int);
__device__ void genAndAdd_device( double *, const uint *, const size_t);
__host__ void genAndAdd_host(double *, const uint *,const size_t);
__host__ __device__ void genAndAdd_generic( double *,const uint *,
                                            const size_t,
                                            const size_t);