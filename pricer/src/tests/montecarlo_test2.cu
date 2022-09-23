#include <iostream>
#include "device_launch_parameters.h"
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_term_structure.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/option_pricer_lib/option_pricer.cuh"
#include "../lib/option_pricer_lib/option_pricer_montecarlo/option_pricer_montecarlo.cuh"
#include "../lib/contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/support_lib/timer_lib/myTimer.cuh"

struct Result
{
    double opt_price;
    double error;
};

__host__ bool run_device(uint * seeds, prcr::Pricer_args * prcr_args,Result * host_results);
void __global__ kernel(uint * seeds, prcr::Pricer_args * prcr_args,Result * dev_results);
bool __host__   simulate_host  (uint* seeds, prcr::Pricer_args* prcr_args, Result* dev_res);
void __device__ simulate_device(uint* seeds, prcr::Contract_eq_option_vanilla * contr_opt, 
                                prcr::Pricer_args * prcr_args, Result * dev_res); 
void __host__ __device__ 
                simulate_generic(uint * seeds, size_t index, 
                                 prcr::Contract_eq_option_vanilla * contr_opt,
                                 prcr::Pricer_args * prcr_args,
                                 Result *  results);

__host__ bool 
run_device(uint * seeds, prcr::Pricer_args * prcr_args,Result * host_res)
{   
    using namespace prcr;
    cudaError_t cudaStatus;
    uint        * dev_seeds;
    Result      * dev_res;
    Pricer_args * dev_prcr_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS * TPB * 4 * sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_prcr_args,sizeof(dev_prcr_args));
    if(cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n");  }

    cudaStatus = cudaMalloc((void**)&dev_res, NBLOCKS * TPB * sizeof(Result));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }



    cudaStatus = cudaMemcpy(dev_seeds, seeds, NBLOCKS * TPB * 4 * sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMemcpy(dev_prcr_args,prcr_args, sizeof(prcr_args),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMemcpy(dev_res, host_res, NBLOCKS*TPB*sizeof(Result), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));


    kernel <<< NBLOCKS, TPB>>>(dev_seeds,dev_prcr_args,dev_res);

    cudaStatus = cudaMemcpy(host_res, dev_res, NBLOCKS*TPB*sizeof(Result), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy4 failed!\n"); }
    fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));


    cudaFree(dev_seeds);
    cudaFree(dev_res);
    cudaFree(dev_prcr_args);

    return cudaStatus;
}




__global__ void 
kernel(uint * seeds, prcr::Pricer_args * prcr_args,Result * dev_results)
{
    using namespace prcr;

    Volatility_surface * volatility_surface = new Volatility_surface(prcr_args->vol_args.vol);

    Yield_curve_flat * yield_curve = new Yield_curve_flat(
                                          prcr_args->eq_descr_args.currency,
                                          prcr_args->yc_args.rate);

    Equity_description * descr = new Equity_description(
                                    prcr_args->eq_descr_args.isin_code,
                                    prcr_args->eq_descr_args.name,
                                    prcr_args->eq_descr_args.currency,
                                    prcr_args->eq_descr_args.dividend_yield,
                                    yield_curve,
                                    volatility_surface);
    
    Equity_prices * starting_point = new Equity_prices(
                                    prcr_args->eq_price_args.price,
                                    prcr_args->eq_price_args.price,
                                    descr);

    Schedule * schedule = new Schedule(
                                    prcr_args->schedule_args.t_ref,
                                    prcr_args->schedule_args.deltat,
                                    prcr_args->schedule_args.dim);
    
    Contract_eq_option_vanilla * eq_option = new Contract_eq_option_vanilla(
                                    starting_point,
                                    schedule,
                                    prcr_args->contract_args.strike_price,
                                    prcr_args->contract_args.contract_type);


    simulate_device(seeds,eq_option,prcr_args, dev_results);

    delete(volatility_surface);
    delete(yield_curve);
    delete(descr);
    delete(starting_point);
    delete(schedule);
    delete(eq_option);
}


__host__ bool 
simulate_host(uint* seeds, prcr::Pricer_args* prcr_args, Result* host_res)
{
    using namespace prcr;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;

    Volatility_surface* volatility_surface = new Volatility_surface(prcr_args->vol_args.vol);

    Yield_curve_flat* yield_curve = new Yield_curve_flat(
        prcr_args->eq_descr_args.currency,
        prcr_args->yc_args.rate);

    Equity_description* descr = new Equity_description(
        prcr_args->eq_descr_args.isin_code,
        prcr_args->eq_descr_args.name,
        prcr_args->eq_descr_args.currency,
        prcr_args->eq_descr_args.dividend_yield,
        yield_curve,
        volatility_surface);

    Equity_prices* starting_point = new Equity_prices(
        prcr_args->eq_price_args.time,
        prcr_args->eq_price_args.price,
        descr);

    Schedule* schedule = new Schedule(
        prcr_args->schedule_args.t_ref,
        prcr_args->schedule_args.deltat,
        prcr_args->schedule_args.dim);

    Contract_eq_option_vanilla* contr_opt = new Contract_eq_option_vanilla(
        starting_point,
        schedule,
        prcr_args->contract_args.strike_price,
        prcr_args->contract_args.contract_type);

    for(int index = 0; index < NBLOCKS*TPB; ++index )
    {
        simulate_generic(seeds, index, contr_opt, prcr_args,host_res);

    }

    delete(volatility_surface);
    delete(yield_curve);
    delete(descr);
    delete(starting_point);
    delete(schedule);
    delete(contr_opt);
    return true; // da mettere giù meglio
}


__device__ void
simulate_device(uint * seeds, 
                prcr::Contract_eq_option_vanilla * contr_opt,
                prcr::Pricer_args * prcr_args,
                Result         * results)
{
    size_t index   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB     = prcr_args->dev_opts.N_threads;
    if (index < NBLOCKS * TPB) simulate_generic(seeds, index, contr_opt,prcr_args, results);    
}

__host__ __device__ void 
simulate_generic(uint * seeds, size_t index, 
                 prcr::Contract_eq_option_vanilla * contr_opt, 
                 prcr::Pricer_args * prcr_args,
                 Result * results)
{
    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];
    size_t PPT = prcr_args->mc_args.N_simulations;

    rnd::GenCombined               * gnr_in  = new rnd::GenCombined(seed0, seed1, seed2, seed3);
    prcr::Process_eq_lognormal     * process = new prcr::Process_eq_lognormal(gnr_in,false,1);
    prcr::Option_pricer_montecarlo * pric    = new prcr::Option_pricer_montecarlo(contr_opt,process,PPT);
    results[index].opt_price = pric->Get_price();
    results[index].error     = pric->Get_price_square();//MC_error();

    delete(gnr_in);
    delete(process);
    delete(pric);
}




int main(int argc, char ** argv)
{
    using namespace prcr;
    srand(time(NULL));
    
    
    std::string filename = "./data/infile_MC_test2.txt";
    Pricer_args * prcr_args = new Pricer_args;
    ReadInputOption(filename,prcr_args);
    
    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB    = prcr_args->dev_opts.N_threads;
    size_t PPT = prcr_args->mc_args.N_simulations;
    //seeeds generation
    uint* seeds = new uint[4 * NBLOCKS * TPB];
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
        seeds[inc] = rnd::genSeed(true);
    //results
    Result* host_res = new Result[NBLOCKS * TPB];
    for(size_t inc = 0; inc < NBLOCKS*TPB; inc ++)
    {
        host_res[inc].opt_price = 0;
        host_res[inc].error = 0;
    }

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;

    if(GPU == true){ 
	Timer gpu_timer;
        status = status && run_device(seeds,prcr_args,host_res);
	gpu_timer.Stop();
    }
    
    if(CPU == true){
        status = status && simulate_host(seeds,prcr_args,host_res);

        double final_error = 0;
	    double squares_sum = 0;
        double final_price = 0;
        for( int i = 0 ; i < NBLOCKS* TPB; ++i)
        {
            //final_error += host_res[i].error;
            final_price += host_res[i].opt_price; 
    	    squares_sum += host_res[i].error;
	    
        }
        final_price /= static_cast<double>(NBLOCKS*TPB);
        //final_error /= static_cast<double>(NBLOCKS*TPB);
	    final_error = compute_final_error(squares_sum, final_price, NBLOCKS*TPB*PPT);
        std::cout << " CPU simulation final results:         \n"
                  << "         - price: " << final_price << "\n"
                  << "         - error: " << final_error << "\n";
    }

    delete[](host_res);
    delete[](seeds);
    delete(prcr_args);
    return status;
}











