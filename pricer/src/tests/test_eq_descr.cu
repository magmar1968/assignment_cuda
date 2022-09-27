#include "header.cuh"
//#include <string>


__host__             bool run_device( prcr::Pricer_args *,prcr::Eq_descr_args *);

__global__           void kernel(prcr::Pricer_args *, prcr::Eq_descr_args *);

__host__             bool simulate_host(prcr::Pricer_args *,prcr::Eq_descr_args *);

__device__           void simulate_device(prcr::Pricer_args *,prcr::Eq_descr_args *);

__host__ __device__  void simulate_generic(prcr::Pricer_args*, prcr::Eq_descr_args*, size_t);


__host__ bool
run_device(prcr::Pricer_args * prcr_args, prcr::Eq_descr_args * host_eq_descr_args)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Pricer_args * dev_prcr_args;
    Eq_descr_args    * dev_eq_descr_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args,sizeof(Pricer_args));
    if(cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n");}

    cudaStatus = cudaMalloc((void**)&dev_eq_descr_args, NBLOCKS * TPB * sizeof(Eq_descr_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    
    cudaStatus = cudaMemcpy(dev_prcr_args,prcr_args, sizeof(Pricer_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); 
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}

    cudaStatus = cudaMemcpy(dev_eq_descr_args,host_eq_descr_args,  NBLOCKS*TPB*sizeof(Eq_descr_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); 
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}

    
    kernel <<< NBLOCKS, TPB>>>(dev_prcr_args,dev_eq_descr_args);

    cudaStatus = cudaMemcpy(host_eq_descr_args, dev_eq_descr_args, NBLOCKS*TPB*sizeof(Eq_descr_args), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n");
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));}   
    
    bool kernel_error_check = true;
    for (int i = 0; i < NBLOCKS * TPB; i++)
    {
        kernel_error_check = kernel_error_check && 
                                            (strcmp(host_eq_descr_args[i].isin_code, prcr_args->eq_descr_args.isin_code) == 0);
                                                                                        //check che memcpy abbia conservato il valore originale
        kernel_error_check = kernel_error_check && (host_eq_descr_args[i].dividend_yield == 2.); 
                                                            //check che simualte_generic abbia lavorato correttamente
    }
    	

    cudaFree(dev_prcr_args);
    cudaFree(dev_eq_descr_args);

    return ((cudaStatus == cudaSuccess) && (kernel_error_check));
}

__global__ void 
kernel(prcr::Pricer_args * prcr_args, prcr::Eq_descr_args * eq_descr_args)
{
    simulate_device(prcr_args, eq_descr_args);
}

__host__   bool simulate_host(prcr::Pricer_args* prcr_args, prcr::Eq_descr_args* eq_descr_args)
{
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    for (int index = 0; index < NBLOCKS * TPB; index++)
    {
        simulate_generic(prcr_args, eq_descr_args, index);
    }


    bool stat = true;
    for (int j = 0; j < NBLOCKS * TPB; j++)
    {
        stat = stat && (strcmp(eq_descr_args[j].isin_code, prcr_args->eq_descr_args.isin_code)==0); //check superfluo su cpu
        stat = stat && (eq_descr_args[j].dividend_yield == 2.); //check che simualte_generic abbia lavorato correttamente
    }
    return stat;
    

}
__device__ void simulate_device(prcr::Pricer_args* prcr_args, prcr::Eq_descr_args* eq_descr_args)
{
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < NBLOCKS * TPB)
    {
        simulate_generic(prcr_args, eq_descr_args, index);
    }
}

__host__ __device__ void simulate_generic(prcr::Pricer_args* prcr_args, prcr::Eq_descr_args* eq_descr_args, size_t index)
{
    eq_descr_args[index].dividend_yield = 2.;   //scrittura in isin code, per controllare che memcpy abbia funzionato (device)
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

    Eq_descr_args * host_eq_descr_args = new Eq_descr_args[NBLOCKS*TPB];

    for(size_t ind = 0; ind < NBLOCKS*TPB; ind ++)
    {
        strcpy(host_eq_descr_args[ind].isin_code, prcr_args->eq_descr_args.isin_code);
        strcpy(host_eq_descr_args[ind].name, prcr_args->eq_descr_args.name);
        strcpy(host_eq_descr_args[ind].currency, prcr_args->eq_descr_args.currency);
        host_eq_descr_args[ind].dividend_yield = prcr_args->eq_descr_args.dividend_yield;


    }

    if(GPU == true)
        for(int i = 0; i < 10000; ++i)
            status = status && (run_device(prcr_args, host_eq_descr_args) == true); //run device returns true if everything is fine
    
    if(CPU == true)
        for(int i = 0; i < 10000; ++i) 
            status = status && simulate_host(prcr_args, host_eq_descr_args);
    
    if (status == true)
        std::cout << "No errors encountered" << std::endl;
    if (status == false)
        std::cout << "An error was encountered" << std::endl;
    //printf("%d\t", strcmp(host_eq_descr_args[0].isin_code, prcr_args->eq_descr_args.isin_code));
    //std::cout << prcr_args->eq_descr_args.isin_code << std::endl;
    delete(host_eq_descr_args);      
    delete(prcr_args);
    return !status;
}
