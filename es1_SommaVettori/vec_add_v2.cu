#include <fstream>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lib.h"





struct Device
{
    bool GPU = 0;
    bool CPU = 0;
};


//somma definita con if
__global__ void gpuArraySum(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //check for overflow
    if (tid < n) {
        c[tid] = 0;
        for (int p = 0; p < 500; p++) {
            c[tid] += exp(a[tid]) + exp(b[tid]);
        }
    }
}


//funzione su GPU definita con while, per scegliere manualmente block_size e numero blocchi
/*
__global__ void gpuArraySum(float* a, float* b, float* c, int n) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //check for overflow
    while (tid < n) {
       c[tid] = 0;
       for (int p = 0; p < 750; p++) {
            c[tid] += exp(a[tid]) + exp(b[tid]);
        }
        tid += blockDim.x*gridDim.x;
    }
}*/


void cpuArraySum(float* a, float* b, float* c, int n)
{
    for (int i = 0; i < n; ++i)
        {
            c[i]=0;
            for (int p=0;p<750;p++)
            {
                c[i] += a[i] + b[i];
            }
        }
}


int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
    typedef const int cint;

    Device dev;
    // parse input
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
    std::string filename;
    if (cmdOptionExists(argv, argv + argc, "-f"))
        filename = getCmdOption(argv, argv + argc, "-f");
    else
        filename = "timeseries.txt";
    


    std::fstream ofs(filename, std::fstream::out);

    // definition of the problem variable
    cint min_size = 100000;
    cint max_size = 120000;
    cint step = 100;
    cint block_size = 256;
    cint iteration = 20;

    //intialize the random engine
    std::random_device rnd;
    std::default_random_engine eng(rnd());

    float* devA, * devB, * devC;


    for (int N = min_size; N < max_size; N += step)
    {
        double time = 0;
        float* A =new float[N];
        float* B =new float[N];
        float* C =new float[N];
        //create the space inside the  GPU memory


        if (dev.GPU)
        {
            cudaStatus = cudaSetDevice(0);
            if (cudaStatus != cudaSuccess)
                {fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");}


            cudaStatus = cudaMalloc((void**)&devA, N * sizeof(float));
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); }
            cudaStatus = cudaMalloc((void**)&devB, N * sizeof(float));
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); }
            cudaStatus = cudaMalloc((void**)&devC, N * sizeof(float));
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); }
        }

        cint grid_size = ((N + block_size) / block_size);
        for (int it = 0; it < iteration; ++it)
        {
            //gen vectors

            fillArray(A, N, eng);
            fillArray(B, N, eng);


            //start time    
            Timer myTimer;

            if (dev.GPU) {
                cudaMemcpy(devA, A, N * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(devB, B, N * sizeof(float), cudaMemcpyHostToDevice);


                gpuArraySum <<< grid_size,block_size >>> (devA, devB, devC, N);
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); }

                cudaMemcpy(C, devC, N * sizeof(float), cudaMemcpyDeviceToHost);
            }
            else
                cpuArraySum(A, B, C, N);

            //stop time
            time += myTimer.getTimeDiff();
        }
        //normalize to the number of iteration
        time /= (double)iteration;
        // print on file
        ofs << N << "," << time << "\n";

        // free all the occupied memory
        if (dev.GPU)
            {cudaFree(devA); cudaFree(devB); cudaFree(devC);}
    }


    return 0;
}

