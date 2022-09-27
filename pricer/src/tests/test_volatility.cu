#include "header.cuh"



__host__             bool run_device( prcr::Pricer_args *,prcr::Vol_args *);

__global__           void kernel(prcr::Pricer_args *,prcr::Vol_args *);

__host__             bool simulate_host(prcr::Pricer_args *,prcr::Vol_args *);
__device__           void simulate_device(prcr::Pricer_args *, prcr::Volatility_surface *,prcr::Vol_args *);
__host__ __device__  void simulate_generic();


__host__ bool
run_device(prcr::Pricer_args * prcr_args, prcr::Vol_args * host_vol_args)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Pricer_args * dev_prcr_args;
    Vol_args    * dev_vol_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args,sizeof(dev_prcr_args));
    if(cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n");}

    cudaStatus = cudaMalloc((void**)&dev_vol_args, NBLOCKS * TPB * sizeof(Vol_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    
    cudaStatus = cudaMemcpy(dev_prcr_args,prcr_args, sizeof(prcr_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMemcpy(dev_vol_args,host_vol_args,  NBLOCKS*TPB*sizeof(Vol_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    
    kernel <<< NBLOCKS, TPB>>>(dev_prcr_args,dev_vol_args);

    cudaStatus = cudaMemcpy(host_vol_args, dev_vol_args, NBLOCKS*TPB*sizeof(Vol_args), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));



    cudaFree(dev_prcr_args);
    cudaFree(dev_vol_args);

    return cudaStatus;
}

__global__ void 
kernel(prcr::Pricer_args * prcr_args, prcr::Vol_args * vol_args)
{
    using namespace prcr;

    Volatility_surface * vol_srfc = new Volatility_surface(prcr_args->vol_args.vol);
    simulate_device(prcr_args,vol_srfc,vol_args);

    delete(vol_srfc);
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

    Vol_args * host_vol_args = new Vol_args[NBLOCKS*TPB];

    for(size_t inc = 0; inc < NBLOCKS*TPB; inc ++)
    {
        host_vol_args[inc].vol = 0.;
    }

    if(GPU == true)
        for(int i = 0; i < 10000; ++i)
            status = status && run_device(prcr_args,host_vol_args);
    
    if(CPU == true)
        for(int i = 0; i < 10000; ++i)
            status = status && simulate_host(prcr_args,host_vol_args);
    
        
    delete(prcr_args);
    return status;
}
