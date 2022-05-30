#include <iostream>
#include "../include/stoch.hpp"
#include "../include/myRandom.hpp"

__global__ void kernel(double **, uint *, int, size_t);
__device__ void createPath_device(double**, uint*, int, size_t);
__host__   void createPath_host(double**, uint*, int, size_t);
__host__ __device__ void createPath_generic(double **,uint *,size_t,int,size_t);




__global__ void kernel(double ** paths,
                       uint  *   seeds, 
                       int       dim,
                       size_t    path_len)
{
    createPath_device(paths,seeds, dim , path_len);
}

__device__ void createPath_device(double ** paths,
                                  uint *    seeds,
                                  int       dim,
                                  size_t    path_len)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    createPath_generic(paths, seeds, index, dim,path_len);
}

__host__ void createPath_host(double ** paths,
                              uint * seeds,
                              int dim,
                              size_t path_len)
{
    for(size_t index = 0; index < dim; index ++)
    {
        createPath_generic(paths,seeds,index,dim,path_len);
    }
}


__device__ __host__ void createPath_generic(double ** paths,
                                       uint *    seeds,
                                       size_t    index,
                                       int       dim,
                                       size_t    path_len )
{
    if(index < dim)
    {
        rnd::GenTausworth gnr(seeds[index],TAUSWORTH_1);
        pricer::EulerSolution path_gnr(1.,1.,0.,0.001);

        for(size_t i = 0; i < path_len; ++i)
        {
            paths[index][i] = path_gnr.get_step(gnr.genGaussian());
        }
    }
}

#define NPATH 1024
#define STEPS 1000
int main()
{

    srand(time(NULL));
    cudaError_t cudaStatus;
    uint *  seeds = new uint[NPATH];
    uint * dev_seeds = new uint[NPATH];

 

    for(int i = 0; i < NPATH; ++i)
    {
        seeds[i] = rnd::genSeed(true);
    }

    double ** paths = new double*[NPATH];
    double ** dev_paths;
    for(int i = 0; i < NPATH; ++i)
    {
        paths[i] = new double[STEPS];
    }
    cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_seeds,NPATH*sizeof(int));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); }
    
    cudaMalloc((void**)&dev_paths,NPATH*STEPS*sizeof(double));
    cudaMemcpy(dev_seeds,seeds,NPATH,cudaMemcpyHostToDevice);

    kernel<<<NPATH,1>>>(dev_paths,dev_seeds,NPATH,STEPS);

    cudaMemcpy(paths,dev_paths,NPATH*STEPS*sizeof(double),cudaMemcpyDeviceToHost);


    //statistica
    
    
    
    return 0;

}
