#include <iostream>
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"

__global__ void kernel(double *, size_t, size_t);
__device__ void create_schedule_device(double *, size_t, size_t);
__host__   void create_schedule_host(double* , size_t, size_t);
__host__ __device__ void create_schedule_generic(double *,size_t,size_t, size_t);

__global__ void kernel(double * schedule, size_t size, size_t dim)
{
    create_schedule_device(schedule,size,dim);
}

__device__ void
create_schedule_device(double * schedule, size_t size, size_t dim)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    create_schedule_generic(schedule,size,dim,index);
}

__host__ void 
create_schedule_host(double * schedule,size_t size, size_t dim)
{
    for(size_t index = 0; index < size; index ++)
    {
        create_schedule_generic(schedule,size,dim,index);
    }
}
__device__ __host__ void 
create_schedule_generic(double * schedule, size_t size,size_t dim, size_t index)
{
    if(index < dim)
        Schedule myschedule(schedule,dim);
    return;
}

#define BLOCKSIZE 256

int main(int argc, char ** argv)
{
    const size_t size = 10;
    const size_t dim  = 1000;
    double * schedule = new double[size];
    for(int i = 0; i < size; ++i)
        schedule[i] = (double)i/10.;

    if(prcr::cmdOptionExists(argv,argc + argv, "-cpu")){
        create_schedule_host(schedule, size, dim);
    }
    else{
        int device_count = 10;
        if(cudaGetDeviceCount(&device_count) == 100)
        {
            std::cerr << "ERROR: no cuda device founded\n";
            return -1;
        }
        else
        {
            std::cout << device_count 
                      << " devices avaible\n";

            cudaError_t cudaStatus;
            double * dev_schedule;
            cudaSetDevice(0);
            cudaStatus = cudaMalloc((void**)&dev_schedule,size*sizeof(double));
            if(cudaStatus != cudaSuccess){
                std::cerr << "ERROR: cudaMalloc failed!\n";
                return -2;

            }
            cudaStatus = cudaMemcpy(dev_schedule, schedule, size,cudaMemcpyHostToDevice);
            if(cudaStatus != cudaSuccess){
                std::cerr << "ERROR: cudaMemcpy failed!\n";
                return -2;
            }
            size_t gridsize = ((dim + BLOCKSIZE) / BLOCKSIZE);
            kernel<<< gridsize, BLOCKSIZE>>>(dev_schedule,size,dim);

            
        }
    }
}