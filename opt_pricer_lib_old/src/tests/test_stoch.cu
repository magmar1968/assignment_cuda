#include <iostream>
#include <fstream>
#include "../stoch_proc/stoch_imp/stoch_euler.cuh"
#include "../support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../support_lib/lib.cuh"


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
        pricer::StocProcess_EulerSolution path_gnr(1.,1.,5.,0.01);

        for(size_t i = 0; i < path_len; ++i)
        {
            paths[index][i] = path_gnr.get_step(gnr.genGaussian());
        }
    }
}

#define NPATH 1024
#define STEPS 1000



int main(int argc, char ** argv)
{
    using namespace pricer;
    pricer::Device dev;
    if (cmdOptionExists(argv, argv + argc, "-h")) {
        std::cout << "usage: -[option] [attibute]       \n"
            << "options: -h help                  \n"
            << "         -g select gpu as device  \n"
            << "         -c select cpu as device  \n"
            << "         -f change output filename\n";
    }
    if (cmdOptionExists(argv, argv + argc, "-cpu") ||
        !cmdOptionExists(argv, argv + argc, "-gpu")) {
        dev.CPU = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-gpu")) {
        dev.GPU = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-both")) {
        dev.GPU = true;
        dev.CPU = true;
    }


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
    
    if(dev.GPU)
    {
        cudaSetDevice(0);
        cudaStatus = cudaMalloc((void**)&dev_seeds,NPATH*sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!\n"); }
        
        cudaMalloc((void**)&dev_paths,NPATH*STEPS*sizeof(double));
        cudaMemcpy(dev_seeds,seeds,NPATH,cudaMemcpyHostToDevice);

        kernel<<<NPATH,1>>>(dev_paths,dev_seeds,NPATH,STEPS);

        cudaMemcpy(paths,dev_paths,NPATH*STEPS*sizeof(double),cudaMemcpyDeviceToHost);
    }
    else
    {
        std::cout << "host path generation selected\n";
        createPath_host(paths,seeds,NPATH,STEPS);
        std::ofstream ofs("./data/test_stoch_paths.txt",std::fstream::out);
        for(int step = 0; step < STEPS; ++step)
        {
            for( int path = 0; path < NPATH; ++path)
            {
                ofs<< paths[path][step] << " ";
            }
            ofs << "\n";
        }
    }


    //statistica
    
    
    
    return 0;

}
