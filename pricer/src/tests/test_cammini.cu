#include <iostream>
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/support_lib/timer_lib/myTimer.cuh"

#define NSIM 1000

bool __host__ run_device        (uint* seeds, prcr::Pricer_args* prcr_args, double* host_last_steps);
void __global__ kernel          (uint* seeds, prcr::Pricer_args* prcr_args, double* dev_last_steps);
bool __host__   simulate_host   (uint* seeds, prcr::Pricer_args* prcr_args, double* host_last_steps);
void __device__ simulate_device (uint* seeds, prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, double* dev_last_steps);
void __host__ __device__ simulate_generic
                                (uint* seeds, size_t, prcr::Pricer_args* , prcr::Equity_prices*, prcr::Schedule*, double*);

__host__ bool
run_device(uint* seeds, prcr::Pricer_args* prcr_args, double* host_last_steps)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    uint* dev_seeds;
    double* dev_last_steps;
    Pricer_args* dev_prcr_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS * TPB * 4 * sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(dev_prcr_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_last_steps, NBLOCKS * TPB * sizeof(double));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }



    cudaStatus = cudaMemcpy(dev_seeds, seeds, NBLOCKS * TPB * 4 * sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMemcpy(dev_prcr_args, prcr_args, sizeof(prcr_args), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMemcpy(dev_last_steps, host_last_steps, NBLOCKS * TPB * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));


    kernel << < NBLOCKS, TPB >> > (dev_seeds, dev_prcr_args, dev_last_steps);

    cudaStatus = cudaMemcpy(host_last_steps, dev_last_steps, NBLOCKS * TPB * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy4 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));


    cudaFree(dev_seeds);
    cudaFree(dev_last_steps);
    cudaFree(dev_prcr_args);

    return cudaStatus;
}




__global__ void
kernel(uint* seeds, prcr::Pricer_args* prcr_args, double* dev_last_steps)
{
    using namespace prcr;

    Equity_description* descr = new Equity_description(
        prcr_args->eq_descr_args.dividend_yield,
        prcr_args->eq_descr_args.yield,
        prcr_args->eq_descr_args.volatility);

    Equity_prices* starting_point = new Equity_prices(
        prcr_args->eq_price_args.time,
        prcr_args->eq_price_args.price,
        descr);

    Schedule* schedule = new Schedule(
        prcr_args->schedule_args.t_ref,
        prcr_args->schedule_args.deltat,
        prcr_args->schedule_args.dim);



    simulate_device(seeds, prcr_args, starting_point, schedule, dev_last_steps);

   
    delete(descr);
    delete(starting_point);
    delete(schedule);
}


__host__ bool
simulate_host(uint* seeds, prcr::Pricer_args* prcr_args, double* last_steps)
{
    using namespace prcr;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    Equity_description* descr = new Equity_description(
        prcr_args->eq_descr_args.dividend_yield,
        prcr_args->eq_descr_args.yield,
        prcr_args->eq_descr_args.volatility);

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
        simulate_generic(seeds, index, prcr_args, starting_point, schedule, last_steps);
    }

    
    delete(descr);
    delete(starting_point);
    delete(schedule);
    return true; // da mettere giù meglio
}


__device__ void
simulate_device(uint* seeds,
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    double* last_steps)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    if (index < NBLOCKS * TPB) simulate_generic(seeds, index, prcr_args, starting_point, schedule, last_steps);
}

__host__ __device__ void
simulate_generic(uint* seeds, size_t index,
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    double* last_steps)
{
    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];
    size_t PPT = prcr_args->mc_args.N_simulations;

    rnd::MyRandomDummy* gnr_in = new rnd::MyRandomDummy();
    prcr::Process_eq_lognormal* process = new prcr::Process_eq_lognormal(gnr_in);
    prcr::Path* path = new prcr::Path(starting_point, schedule, process);
    double length = schedule->Get_dim();
    last_steps[index] = path->Get_equity_prices(length - 1)->Get_eq_price().get_number();

    delete(path);
    delete(gnr_in);
    delete(process);
}




int main(int argc, char** argv)
{
    using namespace prcr;
    double exact_value = 100;   //capire da dove lo vogliamo ricavare
    bool results[NSIM];         //array di bool che registrano se le simulazioni sono andate a buon fine


    for(int sim = 0; sim < NSIM; sim++)
    {



        srand(time(NULL));


        std::string filename = "./data/infile_MC_test2.txt";
        Pricer_args* prcr_args = new Pricer_args;
        ReadInputOption(filename, prcr_args);

        size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
        size_t TPB = prcr_args->dev_opts.N_threads;
        size_t PPT = prcr_args->mc_args.N_simulations;
        //seeeds generation
        uint* seeds = new uint[4 * NBLOCKS * TPB];
        for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
            seeds[inc] = rnd::genSeed(true);
        //last_steps
        double* last_steps = new double[NBLOCKS * TPB];   //array che contiene i valori del prezzo all'ultimo step, per ogni thread
        for (size_t inc = 0; inc < NBLOCKS * TPB; inc++)
        {
            last_steps[inc] = 0;
        }

        bool GPU = prcr_args->dev_opts.GPU;
        bool CPU = prcr_args->dev_opts.CPU;
        bool status = true;

        if (GPU == true) 
        {
            Timer gpu_timer;
            status = status && run_device(seeds, prcr_args, last_steps);
            gpu_timer.Stop();
        }

        if (CPU == true) 
        {
            Timer cpu_timer;
            status = status && simulate_host(seeds, prcr_args, last_steps);
            cpu_timer.Stop();
        }
        
        bool last_step_check = true;

        for (int j = 0; j < NBLOCKS * TPB; j++)
        {
            double delta = abs(last_steps[j] - exact_value);
            last_step_check = last_step_check && (delta < std::pow(10, -12));
        }
        delete[](last_steps);
        delete[](seeds);
        delete(prcr_args);
        
        results[sim] = !(last_step_check && status);  // 0 se tutto va bene
       
    }

    for (int sim = 0; sim < NSIM; sim++)
    {
        if (results[sim] == 1)
            return 1;
    }
    return 0;
        
}











