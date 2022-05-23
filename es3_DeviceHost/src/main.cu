#include "myRandom.hpp"


#define GPU 0

__global__ void kernel(  double *C,const uint *seeds, int dim) {
    genAndAdd_device(C, seeds, dim);
}

__device__ void genAndAdd_device( double *C, const uint *seeds, int dim)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    genAndAdd_generic(C, seeds, index);
}

__host__ __device__ void genAndAdd_generic( double *C,const uint * seeds, const size_t index)
{
    rnd::GenTausworth gnr(seeds[index],TAUSWORTH_1);

    double a = gnr.genUniform();
    double b = gnr.genUniform();

    C[index] = exp(a) + exp(b); 
    
}



int main()
{

}



