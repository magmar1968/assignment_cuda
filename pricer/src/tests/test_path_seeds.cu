#include "header.cuh"
#include <iomanip>


bool __host__   run_device      (prcr::Pricer_args* prcr_args, double* host_last_steps, uint * seeds);
void __global__ kernel          (prcr::Pricer_args* prcr_args, double* dev_last_steps , uint * seeds);
bool __host__   simulate_host   (prcr::Pricer_args* prcr_args, double* host_last_steps, uint * seeds);
void __device__ simulate_device (prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, double*, uint *);
void __host__ __device__ simulate_generic
                                (size_t, prcr::Pricer_args* , prcr::Equity_prices*, prcr::Schedule*, double*, uint *);

__host__ bool
run_device(prcr::Pricer_args* prcr_args, double* host_last_steps, uint * host_seeds)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    double* dev_last_steps;
    Pricer_args* dev_prcr_args;
    uint* dev_seeds;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(Pricer_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_last_steps, NBLOCKS * TPB * sizeof(double));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_seeds,NBLOCKS*TPB*4*sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc4 failed!\n"); }


    cudaStatus = cudaMemcpy(dev_prcr_args, prcr_args, sizeof(Pricer_args), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){ 
        fprintf(stderr, "cudaMemcpy1 failed!\n"); 
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMemcpy(dev_last_steps, host_last_steps, NBLOCKS * TPB * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){ 
        fprintf(stderr, "cudaMemcpy2 failed!\n"); 
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMemcpy(dev_seeds, host_seeds, NBLOCKS * TPB *4* sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){ 
        fprintf(stderr, "cudaMemcpy3 failed!\n"); 
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }



    kernel << < NBLOCKS, TPB >> > (dev_prcr_args, dev_last_steps,dev_seeds);

    cudaStatus = cudaMemcpy(host_last_steps, dev_last_steps, NBLOCKS * TPB * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){ 
        fprintf(stderr, "cudaMemcpy4 failed!\n"); 
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_last_steps);
    cudaFree(dev_prcr_args);
    cudaFree(dev_seeds);

    return cudaStatus;
}




__global__ void
kernel(prcr::Pricer_args* prcr_args, double* dev_last_steps, uint * dev_seeds)
{
    using namespace prcr;

    Equity_description* descr = new Equity_description(
        prcr_args->eq_descr_args.dividend_yield,
        prcr_args->eq_descr_args.rate,
        prcr_args->eq_descr_args.vol);

    Equity_prices* starting_point = new Equity_prices(
        prcr_args->eq_price_args.time,
        prcr_args->eq_price_args.price,
        descr);

    Schedule* schedule = new Schedule(
        prcr_args->schedule_args.t_ref,
        prcr_args->schedule_args.deltat,
        prcr_args->schedule_args.dim);



    simulate_device( prcr_args, starting_point, schedule, dev_last_steps, dev_seeds);

   
    delete(descr);
    delete(starting_point);
    delete(schedule);
}


__host__ bool
simulate_host(prcr::Pricer_args* prcr_args, double* last_steps,uint * host_seeds)
{
    using namespace prcr;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    Equity_description* descr = new Equity_description(
        prcr_args->eq_descr_args.dividend_yield,
        prcr_args->eq_descr_args.rate,
        prcr_args->eq_descr_args.vol);

    Equity_prices* starting_point = new Equity_prices(
        prcr_args->eq_price_args.time,
        prcr_args->eq_price_args.price,
        descr);

    Schedule* schedule = new Schedule(
        prcr_args->schedule_args.t_ref,
        prcr_args->schedule_args.deltat,
        prcr_args->schedule_args.dim);


    for (int index = 0; index < NBLOCKS * TPB; ++index)
    {
        simulate_generic(index, prcr_args, starting_point, schedule, last_steps, host_seeds);
    }

    
    delete(descr);
    delete(starting_point);
    delete(schedule);
    return true; // da mettere gi� meglio
}


__device__ void
simulate_device(
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    double* last_steps,
    uint * dev_seeds)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = gridDim.x;
    size_t TPB = blockDim.x;
    if (index < NBLOCKS * TPB) simulate_generic(index, prcr_args, starting_point, schedule, last_steps,dev_seeds);
}

__host__ __device__ void
simulate_generic(size_t index,
    prcr::Pricer_args*   prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule*      schedule,
    double*              last_steps,
    uint*                seeds)
{

    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];

    rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0,seed1,seed2,seed3);
    prcr::Process_eq_lognormal* process 
                = new prcr::Process_eq_lognormal(gnr_in,prcr_args->stc_pr_args.exact);

                
    prcr::Path path(starting_point, schedule, process);
    last_steps[index] = path.Get_last_eq_price();
    delete(process);
    delete(gnr_in);
}




int main(int argc, char** argv)
{
    using namespace prcr;


    std::string filename = "./data/infile_test_regen_path_seeds.txt";
    Pricer_args* prcr_args = new Pricer_args;
    ReadInputOption(filename, prcr_args);

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    // gen seeds 
    srand(time(NULL));
    uint* seeds = new uint[4 * NBLOCKS * TPB];
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++){   
        seeds[inc] = rnd::genSeed(true);
        if(inc < 10){
            std::cout << "seed[" << inc << "]: " << seeds[inc] << std::endl;
        }
    }


    

    //last_steps
    double * last_steps_gpu = new double[NBLOCKS * TPB];   //array che contiene i valori del prezzo all'ultimo step, per ogni thread
    double * last_steps_cpu = new double[NBLOCKS * TPB];
    for (size_t inc = 0; inc < NBLOCKS * TPB; inc++)
    {
        last_steps_gpu[inc] = 0;
        last_steps_cpu[inc] = 0;
    }

    bool last_step_check_cpu = true;
    bool last_step_check_gpu = true;
    double exact_value = 0.;

    if(prcr_args->stc_pr_args.exact == true){
        exact_value = 109.417428370521;
    }else
        exact_value = 117.318876218338;

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;
    
    if (GPU == true) 
    {
        for(int i = 0; i < N_TEST_SIM; ++i)
            status = status && run_device(prcr_args, last_steps_gpu,seeds);

        double dev_std = prcr::dev_std(last_steps_gpu,NBLOCKS*TPB);
        double mean    = prcr::avg(last_steps_gpu,NBLOCKS*TPB); 
        double delta   = abs(mean - exact_value);

        if(delta > 3* dev_std){
            std::cerr << "ERROR: something went wrong. Average of the results is outside the 3 sigma\n"
                      << "       range from the exact value. Mean: " << mean << std::endl;
        
        }
    }
    if (CPU == true) 
    {
        for(int i = 0; i < N_TEST_SIM; ++i)
            status = status && simulate_host(prcr_args, last_steps_cpu,seeds);
        

        double dev_std = prcr::dev_std(last_steps_cpu,NBLOCKS*TPB);
        double mean    = prcr::avg(last_steps_cpu,NBLOCKS*TPB); 
        double delta   = abs(mean - exact_value);

        if(delta > 3* dev_std){
            std::cerr << "ERROR: something went wrong. Average of the results is outside the 3 sigma\n"
                      << "       range from the exact value. Mean: " << mean << std::endl;
        }

        std::cout << "mean: " << std::setprecision(12) << mean << std::endl;
    }


    delete[](last_steps_cpu);
    delete[](last_steps_gpu);
    delete(prcr_args);
    

    if(last_step_check_cpu && last_step_check_gpu){
        std::cout << "no error encountered" << std::endl;
        return 0;
    }
    else if (last_step_check_cpu == false and last_step_check_gpu == true){
        std::cerr << "ERROR: cpu simulation failed!" << std::endl;
        return 1;
    }
    else if (last_step_check_cpu == true and last_step_check_gpu == false){
        std::cerr << "ERROR: gpu simulation failed!" << std::endl;
        return 2;
    }
    else{
        std::cerr << "ERROR: both cpu and gpu simulations failed! " << std::endl;
        return 3;
    }
}
