#include "header.h"
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
    uint   * devSeeds;
    double * C = new double[N];
    uint * seeds = new uint[N];
    for(size_t i = 0; i < N; ++i )
    {
        seeds[i] = rnd::genSeed(true);
    } 



    if(GPU == true){
        uint grid_size = ((N + BLOCK_SIZE)/BLOCK_SIZE);
        cudaMalloc((void**)&devC,N*sizeof(double));
        cudaMalloc((void**)&devSeeds, N * sizeof(uint));

        cudaMemcpy(devSeeds,seeds,N*sizeof(uint),cudaMemcpyHostToDevice);

        kernel <<< grid_size, BLOCK_SIZE >>> (devC,seeds,N);
        
        cudaMemcpy(C,devC,N*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; ++i)
        {
            std::cout << C[i] << std::endl;
        }
    }
    else{

    }

}



