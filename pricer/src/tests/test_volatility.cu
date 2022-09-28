#include "header.cuh"



__host__             bool  run_device( prcr::Pricer_args *,prcr::Vol_args *);

__global__           void kernel(prcr::Pricer_args *,prcr::Vol_args *);

__host__             bool  simulate_host(prcr::Pricer_args *,prcr::Vol_args *);
__device__           void simulate_device(prcr::Pricer_args *, prcr::Volatility_surface *,prcr::Vol_args *);
__host__ __device__  void simulate_generic(size_t index, prcr::Volatility_surface *,prcr::Vol_args *);


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
    
    cudaStatus = cudaMemcpy(dev_vol_args,host_vol_args,  NBLOCKS*TPB*sizeof(Vol_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
    

    
    kernel <<< NBLOCKS, TPB>>>(dev_prcr_args,dev_vol_args);

    cudaStatus = cudaMemcpy(host_vol_args, dev_vol_args, NBLOCKS*TPB*sizeof(Vol_args), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }

    bool kernel_error_check = true;
    for (int i = 0; i < NBLOCKS*TPB; i++)
    {
	    kernel_error_check = kernel_error_check && (host_vol_args[i].vol == prcr_args->vol_args.vol);  //controlla che simulate_generic abbia 										  //agito correttamente
        if(!kernel_error_check)
        {
            std::cerr << "something is going wrong\n";
        }
    }

    cudaFree(dev_prcr_args);
    cudaFree(dev_vol_args);

    return ((cudaStatus==cudaSuccess) && (kernel_error_check));
}

__global__ void 
kernel(prcr::Pricer_args * prcr_args, prcr::Vol_args * vol_args)
{
    using namespace prcr;

    Volatility_surface * vol_srfc = new Volatility_surface(prcr_args->vol_args.vol);
    simulate_device(prcr_args,vol_srfc,vol_args);

    delete(vol_srfc);
}

__device__ void
simulate_device(prcr::Pricer_args        * prcr_args, 
                prcr::Volatility_surface * vol_sfrc,
                prcr::Vol_args           * vol_args)
{
    size_t index   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB     = prcr_args->dev_opts.N_threads;
    if (index < NBLOCKS * TPB) simulate_generic( index, vol_sfrc,vol_args);
}

__host__ bool
simulate_host(prcr::Pricer_args * prcr_args,
              prcr::Vol_args    * vol_args)
{
    using namespace prcr;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    Volatility_surface * vol_srfc = new Volatility_surface(prcr_args->vol_args.vol);


    for (int index = 0; index < NBLOCKS * TPB; index++){
        simulate_generic(index,vol_srfc,vol_args);
    }
    bool stat = true;
    for (int j = 0; j < NBLOCKS * TPB; j++)
    {
        stat = stat && (vol_args[j].vol == prcr_args->vol_args.vol); //controllo che simulate generic abbia sovrascritto correttamente il campo
    }
    return stat;
}


__device__ __host__ void
simulate_generic(size_t index,
                 prcr::Volatility_surface * vol_sfrc,
                 prcr::Vol_args *           vol_args)
{
    vol_args[index].vol = vol_sfrc->Get_volatility();
}

int main(int argc, char ** argv)
{
    using namespace prcr;
    std::string filename = "./data/infile_MC_test2.txt";
    Pricer_args * prcr_args = new Pricer_args;
    ReadInputOption(filename,prcr_args);

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    Vol_args * host_vol_args = new Vol_args[NBLOCKS*TPB];

    for(size_t inc = 0; inc < NBLOCKS*TPB; inc ++)
    {
        host_vol_args[inc].vol = 0.;
    }
    bool status_gpu = true, status_cpu = true;
    
    if(GPU == true)
        for(int i = 0; i < N_TEST_SIM; ++i)
            status_gpu = status_gpu && run_device(prcr_args,host_vol_args);
    
    for(size_t inc = 0; inc < NBLOCKS*TPB; inc ++)
    {
        host_vol_args[inc].vol = 0.;
    }

    if(CPU == true)
        for(int i = 0; i < N_TEST_SIM; ++i)
            status_cpu = status_cpu && simulate_host(prcr_args,host_vol_args);

    delete(prcr_args);

    if ( (status_gpu && status_cpu) == true){
        std::cout << "No errors encountered" << std::endl;
        return 0;
    }
    else if(status_gpu == false && status_cpu == false)    {
        std::cerr << "ERROR: gpu and cpu simulations didn't work properly\n";
        return -1;
    }
    else if(status_gpu == false && status_cpu == true ){
        std::cerr << "ERROR: gpu simulation didn't work properly\n";
        return -2;
    }
    else{
        std::cerr << "ERROR: cpu simulation didn't work properly\n";
        return -3;
    }

        
}
