#include "header.cuh"



__host__             bool run_device( prcr::Pricer_args *,prcr::Yc_args *);

__global__           void kernel(prcr::Pricer_args *, prcr::Yc_args *);

__host__             bool simulate_host(prcr::Pricer_args *,prcr::Yc_args *);

__device__           void simulate_device(prcr::Pricer_args *,prcr::Yc_args *);

__host__ __device__  void simulate_generic(prcr::Pricer_args*, prcr::Yc_args*, size_t);


__host__ bool
run_device(prcr::Pricer_args * prcr_args, prcr::Yc_args * host_yc_args)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Pricer_args * dev_prcr_args;
    Yc_args    * dev_yc_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args,sizeof(Pricer_args));
    if(cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n");}

    cudaStatus = cudaMalloc((void**)&dev_yc_args, NBLOCKS * TPB * sizeof(Yc_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    
    cudaStatus = cudaMemcpy(dev_prcr_args,prcr_args, sizeof(Pricer_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); 
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}

    cudaStatus = cudaMemcpy(dev_yc_args,host_yc_args,  NBLOCKS*TPB*sizeof(Yc_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); 
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}

    
    kernel <<< NBLOCKS, TPB>>>(dev_prcr_args,dev_yc_args);

    cudaStatus = cudaMemcpy(host_yc_args, dev_yc_args, NBLOCKS*TPB*sizeof(Yc_args), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n");
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}   
    
    bool kernel_error_check = true;
    for (int i = 0; i < NBLOCKS*TPB; i++)
    {
	kernel_error_check = kernel_error_check && (host_yc_args[i].rate == 2.);  //controlla che simulate_generic abbia 										  //agito correttamente
    }	

    cudaFree(dev_prcr_args);
    cudaFree(dev_yc_args);

    return ((cudaStatus==cudaSuccess) && (kernel_error_check));
}

__global__ void 
kernel(prcr::Pricer_args * prcr_args, prcr::Yc_args * yc_args)
{
    simulate_device(prcr_args, yc_args);
}

__host__   bool simulate_host(prcr::Pricer_args* prcr_args, prcr::Yc_args* yc_args)
{
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    for (int index = 0; index < NBLOCKS * TPB; index++)
    {
        simulate_generic(prcr_args, yc_args, index);
    }
    bool stat = true;
    for (int j = 0; j < NBLOCKS * TPB; j++)
    {
        stat = stat && (yc_args[j].rate == 2.); //controllo che simulate generic abbia sovrascritto correttamente il campo
    }
    return stat;
    

}
__device__ void simulate_device(prcr::Pricer_args* prcr_args, prcr::Yc_args* yc_args)
{
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < NBLOCKS * TPB)
    {
        simulate_generic(prcr_args, yc_args, index);
    }
}

__host__ __device__ void simulate_generic(prcr::Pricer_args* prcr_args, prcr::Yc_args* yc_args, size_t index)
{
    yc_args[index].rate = 2.;   //scrittura in yield curve rate, per controllare che memcpy abbia funzionato (device)
}


int main(int argc, char ** argv)
{
    using namespace prcr;
    std::string filename = "./data/infile_MC_test2.txt";
    Pricer_args * prcr_args = new Pricer_args;
    ReadInputOption(filename,prcr_args);

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    Yc_args * host_yc_args = new Yc_args[NBLOCKS*TPB];

    for(size_t inc = 0; inc < NBLOCKS*TPB; inc ++)
    {
        host_yc_args[inc].rate = prcr_args->yc_args.rate;
        host_yc_args[inc].dim = prcr_args->yc_args.dim;
        host_yc_args[inc].structured = prcr_args->yc_args.structured;
    }

    if(GPU == true)
        for(int i = 0; i < 10000; ++i)
            status = status && (run_device(prcr_args, host_yc_args) == true); //run device returns true if everything is fine
    
    if(CPU == true)
        for(int i = 0; i < 10000; ++i) 
            status = status && simulate_host(prcr_args, host_yc_args);
    
    if (status == true)
        std::cout << "No errors encountered" << std::endl;
    if (status == false)
        std::cout << "An error was encountered" << std::endl;
    delete(host_yc_args);       
    delete(prcr_args);
    return !status;
}
