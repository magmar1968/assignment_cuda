#include <iostream>
#include "/../src/myRandom/myRandom.cuh"
#include "/../src/myRandom/myRandom_gnr/tausworth.cuh"
#include "/../src/stoch_proc/stoch_imp/stoch_euler.cuh"
#include "/../src/stoch_proc/stoch.cuh"
#include "/../src/payOff/pathGenerator.cuh"
#include "/../src/payOff/schedule/schedule.cuh"

typedef unsigned int uint;

__global__ void kernel(double**, uint*, pricer::Schedule*, int, size_t);
__device__ void createPath_device(double**, uint*, pricer::Schedule*, int, size_t);
__host__   void createPath_host(double**, uint*, pricer::Schedule*, int, size_t);
__host__ __device__ void createPath_generic(double**, uint*, size_t, pricer::Schedule*, int, size_t);




__global__ void kernel(double** paths,
    uint* seeds,
    pricer::Schedule* cal,
    int       dim,
    size_t    path_len)
{
    createPath_device(paths, seeds, cal, dim, path_len);
}

__device__ void createPath_device(double** paths,
    uint* seeds,
    pricer::Schedule* cal,
    int       dim,
    size_t    path_len)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    createPath_generic(paths, seeds, index, cal, dim, path_len);
}

__host__ void createPath_host(double** paths,
    uint* seeds,
    pricer::Schedule* cal,
    int dim,
    size_t path_len)
{
    for (size_t index = 0; index < dim; index++)
    {
        createPath_generic(paths, seeds, index, cal, dim, path_len);
    }
}


__device__ __host__ void createPath_generic(double** paths,
    uint* seeds,
    size_t    index,
    pricer::Schedule* cal,
    int       dim,
    size_t    path_len)
{
    if (index < dim)
    {
      rnd::GenTausworth gnr(seeds[index], TAUSWORTH_1);
      double dt = 0.001;
      pricer::StocProcess_EulerSolution path_gnr(1., 1., 10., dt);

        pricer::PathImp cammino(&gnr, &path_gnr, cal, path_len);

        for (size_t i = 0; i < path_len; ++i)
        {
           // paths[index][i] = path_gnr.get_step(gnr.genGaussian());
            paths[index][i] = cammino.getPath()[i];
        }
    }
}

#define NPATH 3
#define STEPS 5

int main()
{

    srand(time(NULL));
    cudaError_t cudaStatus;
    uint* seeds = new uint[NPATH];
    uint* dev_seeds = new uint[NPATH];



    for (int i = 0; i < NPATH; ++i)
    {
        seeds[i] = rnd::genSeed(true);
    }



    double** paths = new double* [NPATH];
    double** dev_paths;
    for (int i = 0; i < NPATH; ++i)
    {
        paths[i] = new double[STEPS];
    }

    pricer::Schedule calendar(0, 0.2, 5);

    /*CudaSetDevice(0);
     cudaStatus = cudaMalloc((void**)&dev_seeds, NPATH * sizeof(int));
     if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!\n"); }

     cudaMalloc((void**)&dev_paths, NPATH * STEPS * sizeof(double));
     cudaMemcpy(dev_seeds, seeds, NPATH, cudaMemcpyHostToDevice);

     kernel << <NPATH, 1 >> > (dev_paths, dev_seeds, NPATH, STEPS);

     cudaMemcpy(paths, dev_paths, NPATH * STEPS * sizeof(double), cudaMemcpyDeviceToHost);*/

    createPath_host(paths, seeds, &calendar, NPATH, STEPS);

    //statistica
    double* ptr = new double[STEPS];
    calendar.Get_t(ptr);
    std::cout << "date   ";
    for (int k = 0; k < STEPS; k++)
    {
        std::cout << ptr[k] << " ";
    }
    std::cout << std::endl;
    for (int j = 0; j < NPATH; j++)
    {
        for (int jj = 0; jj < STEPS; jj++)
        {
            std::cout << paths[j][jj] << " ";
        }
        std::cout << std::endl;
    }
    

    return 0;

}
