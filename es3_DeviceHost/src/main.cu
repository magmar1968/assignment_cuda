#include "myRandom.hpp"
#include <iostream>

#define GPU 0
#define N   1000000

// device call
__global__ void kernel(  double *C,const uint *seeds, int dim) {
    genAndAdd_device(C, seeds, dim);
}

__device__ void genAndAdd_device( double *C, const uint *seeds, const size_t dim)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    genAndAdd_generic(C, seeds, index, dim);
}

// host call

__host__ void genAndAdd_host(double *C, const uint *seeds,const size_t dim)
{
    for(size_t index = 0; index < dim; index++)
    {
        genAndAdd_generic(C,seeds,index,dim);
    }
}

__host__ __device__ void genAndAdd_generic( double *C,const uint * seeds, const size_t index, const size_t dim)
{
    if(index < dim)
    {
        rnd::GenTausworth gnr(seeds[index],TAUSWORTH_1);

        double a = gnr.genUniform();
        double b = gnr.genUniform();
        C[index] = exp(a) + exp(b); 
        
    }
    
}

int main()
{
    double * devC;
    double C[N] = {};

    if(GPU == true){
        cudaMalloc((void**)&devC,N*sizeof(double));
    }
    else{

    }

}



