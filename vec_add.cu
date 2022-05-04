#include <fstream>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include "lib.h"

struct Device
{
    bool GPU = 0;
    bool CPU = 0;
};



__global__ void gpuArraySum(float* a, float* b, float* c,int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //check for overflow
    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }
}

void cpuArraySum(float *a, float * b, float*c, int n)
{
    for( int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


int main(int argc, char ** argv)
{
    typedef const int cint;

    Device dev;
    // parse input
    if( cmdOptionExists(argv, argv + argc, "-h")){
        std::cout << "usage: -[option] [attibute]       \n" 
                  << "options: -h help                  \n" 
                  << "         -g select gpu as device  \n" 
                  << "         -c select cpu as device  \n" 
                  << "         -f change output filename\n";
    }
    if( cmdOptionExists(argv, argv + argc, "-cpu") or 
       !cmdOptionExists(argv, argv + argc, "-gpu")){
        dev.CPU = true;
    }
    if( cmdOptionExists(argv, argv + argc, "-gpu")){
        dev.GPU = true;
    }
    std::string filename;
    if( cmdOptionExists(argv, argv + argc, "-f"))
        filename = getCmdOption(argv, argv + argc, "-f");
    else
        filename = "timeseries.txt";

    std::fstream ofs(filename,std::fstream::out);
    
    // definition of the problem variable
    cint min_size   = 1000000;
    cint max_size   = 1000100;
    cint block_size = 256;
    cint iteration  = 10;
   
    //intialize the random engine
    std::random_device rnd;  
    std::default_random_engine eng(rnd());

    float *devA, *devB, *devC;

    std::cerr<< "fino a qui\n";
    int cont = 0;
    for (int N = min_size; N < max_size; ++N)
    {
        double time = 0;
        float A[N] = {};
        float B[N] = {};
        float C[N] = {};
        //create the space inside the  GPU memory
        if(dev.GPU)
        {
            cudaMalloc( (void**)&devA, N*sizeof(float) );
            cudaMalloc( (void**)&devB, N*sizeof(float) );
            cudaMalloc( (void**)&devC, N*sizeof(float) );
        }
        
        cint grid_size = ((N + block_size)/block_size);
        for(int it = 0; it < iteration ; ++it)
        {
            ++ cont;
            std::cerr << "iterazione " <<  cont << std::endl;
            //gen vectors
            fillArray(A,N,eng);
            fillArray(B,N,eng);

            //start time    
            Timer myTimer;

            if(dev.GPU){
                cudaMemcpy( devA, A, N*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy( devB, B, N*sizeof(float), cudaMemcpyHostToDevice);
                gpuArraySum<<<grid_size,block_size>>>(devA, devB, devC,N);
                cudaMemcpy(C, devC, N*sizeof(float), cudaMemcpyDeviceToHost);
            }
            else
                cpuArraySum(A,B,C,N);

            //stop time
            time += myTimer.getTimeDiff();
        }
        //normalize to the number of iteration
        time/= iteration;
        // print on file
        ofs << N << "," << time << "\n";
        
        // free all the occupied memory
        if(dev.GPU)
            cudaFree(devA);cudaFree(devB);cudaFree(devC);
    }


    return 0;
}




