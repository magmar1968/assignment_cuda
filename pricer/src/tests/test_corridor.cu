#include "header.cuh"
#include <iomanip>

#define NSIM 1


struct Result
{
    double p_off;
    double p_off2;
};

bool __host__ run_device(prcr::Pricer_args* prcr_args, Result* host_results,uint *);
void __global__ kernel(prcr::Pricer_args* prcr_args, Result* dev_results, uint *);
bool __host__   simulate_host(prcr::Pricer_args* prcr_args, Result* host_results, uint*);
void __device__ simulate_device(prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, Result* dev_results, uint*);
void __host__ __device__ simulate_generic
(size_t, prcr::Pricer_args*, prcr::Equity_prices*, prcr::Schedule*, Result*, uint*);

__host__ bool
run_device(prcr::Pricer_args* prcr_args, Result* host_results, uint * host_seeds)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Result* dev_results;
    Pricer_args* dev_prcr_args;
    uint * dev_seeds;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(Pricer_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_results, NBLOCKS * TPB * sizeof(Result));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS * TPB *4 * sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }




    cudaStatus = cudaMemcpy(dev_prcr_args, prcr_args, sizeof(Pricer_args), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMemcpy(dev_results, host_results, NBLOCKS * TPB * sizeof(Result), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaMemcpy(dev_seeds, host_seeds, NBLOCKS * TPB * 4 * sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }



    kernel << < NBLOCKS, TPB >> > (dev_prcr_args, dev_results, dev_seeds);

    cudaStatus = cudaMemcpy(host_results, dev_results, NBLOCKS * TPB * sizeof(Result), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy4 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_results);
    cudaFree(dev_prcr_args);
    cudaFree(dev_seeds);

    return cudaStatus;
}




__global__ void
kernel(prcr::Pricer_args* prcr_args, Result* dev_results, uint * dev_seeds)
{
    using namespace prcr;

    Equity_description descr(
        prcr_args->eq_descr_args.dividend_yield,
        prcr_args->eq_descr_args.rate,
        prcr_args->eq_descr_args.vol);

    Equity_prices starting_point(
        prcr_args->eq_price_args.time,
        prcr_args->eq_price_args.price,
        &descr);

    Schedule schedule(
        prcr_args->schedule_args.t_ref,
        prcr_args->schedule_args.deltat,
        prcr_args->schedule_args.dim);

    simulate_device(prcr_args, &starting_point, &schedule, dev_results,dev_seeds);

}


__host__ bool
simulate_host(prcr::Pricer_args* prcr_args, Result* host_results, uint * host_seeds)
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
        simulate_generic(index, prcr_args, starting_point, schedule, host_results,host_seeds);
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
    Result* dev_results,
    uint * dev_seeds)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = gridDim.x;
    size_t TPB = blockDim.x;
    if (index < NBLOCKS * TPB) simulate_generic(index, prcr_args, starting_point, schedule, dev_results,dev_seeds);
}

__host__ __device__ void
simulate_generic(size_t index,
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    Result* results,
    uint * seeds)
{

    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];

    rnd::GenCombined gnr_in(seed0,seed1,seed2,seed3);


    prcr::Process_eq_lognormal process(&gnr_in, prcr_args->stc_pr_args.exact);

    prcr::Contract_eq_option_exotic_corridor contr_opt(starting_point,
                                               schedule,
                                               prcr_args->contract_args.strike_price,
                                               prcr_args->contract_args.contract_type,
                                               prcr_args->contract_args.B,
                                               prcr_args->contract_args.N,
                                               prcr_args->contract_args.K);
    size_t _N = prcr_args->mc_args.N_simulations;
    prcr::Option_pricer_montecarlo pricer(&contr_opt, &process, _N);

    results[index].p_off =  pricer.Get_price();
    results[index].p_off2 = pricer.Get_price_square();

}




int main(int argc, char** argv)
{
    using namespace prcr;

    srand(time(NULL));


    std::string filename = "./data/infile_test_corridor.txt";
    Pricer_args* prcr_args = new Pricer_args;
    ReadInputOption(filename, prcr_args);

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    size_t PPT = prcr_args->mc_args.N_simulations;
    // gen seeds 
    srand(time(NULL));
    uint* seeds = new uint[4 * NBLOCKS * TPB];
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
        seeds[inc] = rnd::genSeed(true);


    //last_steps
    Result* gpu_results = new Result[NBLOCKS * TPB];   //array che contiene i valori del prezzo all'ultimo step, per ogni thread
    Result* cpu_results = new Result[NBLOCKS * TPB];
    for (size_t inc = 0; inc < NBLOCKS * TPB; inc++)
    {
        gpu_results[inc].p_off = 0;
        gpu_results[inc].p_off2 = 0;
        cpu_results[inc].p_off = 0;
        cpu_results[inc].p_off2 = 0;
    }

    bool results_check_cpu = true;
    bool results_check_gpu = true;
    double exact_value = 0.;

    
   exact_value = Evaluate_corridor(     prcr_args->eq_descr_args.rate,
					prcr_args->eq_descr_args.vol,
					prcr_args->schedule_args.dim - 1,
					(prcr_args->schedule_args.dim - 1) * prcr_args->schedule_args.deltat,
                                        prcr_args->contract_args.K,
					prcr_args->contract_args.B);
    
    std::cout << "exact value: "<< exact_value << std::endl;
    std::cout << prcr_args->schedule_args.dim * prcr_args->schedule_args.deltat << std::endl;
    std::cout << prcr_args->schedule_args.dim << std::endl;
    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;

    if (GPU == true)
    {
        for (int i = 0; i < N_TEST_SIM; ++i)
            status = status && run_device(prcr_args, gpu_results,seeds);
        
        
        
        double gpu_squares_sum  = 0.;
        double gpu_final_result = 0.;
        for(size_t i = 0; i < NBLOCKS * TPB; i++)
        {
            gpu_final_result += gpu_results[i].p_off;
            gpu_squares_sum  += gpu_results[i].p_off2;
        }
        gpu_final_result /= double(NBLOCKS * TPB);
        double gpu_MC_error = compute_final_error(gpu_squares_sum, gpu_final_result, NBLOCKS * TPB*PPT);
        std::cout << "gpu result:" << gpu_final_result << std::endl;
        double delta;
        delta = gpu_final_result - exact_value;
        if (abs(delta) > 3 * gpu_MC_error)
        {
            results_check_gpu = false;
            std::cout << "Errore MC: " << gpu_MC_error << std::endl;
            std::cout << "Delta: " << delta << std::endl;
        }
        
    }


    if (CPU == true)
    {
        for (int i = 0; i < N_TEST_SIM; ++i)
            status = status && simulate_host(prcr_args, cpu_results,seeds);
        
        double cpu_squares_sum  = 0.;
        double cpu_final_result = 0.;
        for(size_t i = 0; i < NBLOCKS * TPB; i++)
        {
            cpu_final_result += cpu_results[i].p_off;
            cpu_squares_sum  += cpu_results[i].p_off2;
        }
        cpu_final_result /= double(NBLOCKS * TPB);
        double cpu_MC_error = compute_final_error(cpu_squares_sum, cpu_final_result, NBLOCKS * TPB*PPT);

        std::cout << "cpu result:" << cpu_final_result << std::endl;
        double delta;
        delta = cpu_final_result - exact_value;
        if (abs(delta) > 3 * cpu_MC_error)
        {
            results_check_cpu = false;
            std::cout << "mean: "      << std::setprecision(12) << cpu_final_result << std::endl;
            std::cout << "Errore MC: " << cpu_MC_error << std::endl;
            std::cout << "Delta: "     << delta << std::endl;
        }
    }


    delete[](cpu_results);
    delete[](gpu_results);
    delete(prcr_args);


    if (results_check_cpu && results_check_gpu) {
        std::cout << "no error encountered" << std::endl;
        return 0;
    }
    else if (results_check_cpu == false and results_check_gpu == true) {
        std::cerr << "ERROR: cpu simulation failed!" << std::endl;
        return 1;
    }
    else if (results_check_cpu == true and results_check_gpu == false) {
        std::cerr << "ERROR: gpu simulation failed!" << std::endl;
        return 2;
    }
    else {
        std::cerr << "ERROR: both cpu and gpu simulations failed! " << std::endl;
        return 3;
    }
}
