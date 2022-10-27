#include "header.cuh"
#include <iomanip>


#define MAX_M  120
#define MIN_M  5

struct Result
{
    double p_off  = 0.;
    double p_off2 = 0.;
};

struct MC_result
{
    double value    = 0.;
    double MC_error = 0.;
};

bool __host__ run_device(const prcr::Pricer_args* prcr_args, Result* host_results,const uint *);
void __global__ kernel( prcr::Pricer_args* prcr_args, Result* dev_results, uint *);
bool __host__   simulate_host(const prcr::Pricer_args* prcr_args, Result* host_results, const uint*);
void __device__ simulate_device( prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, Result* dev_results, uint*);
void __host__ __device__ simulate_generic
(size_t, const prcr::Pricer_args*, prcr::Equity_prices*, prcr::Schedule*, Result*,const uint*);
MC_result __host__ compute_MC_result(const prcr::Pricer_args *, const Result *);


__host__ MC_result
compute_MC_result(const prcr::Pricer_args * prcr_args, const Result * results)
{
    MC_result mc_result;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB     = prcr_args->dev_opts.N_threads;
    size_t PPT     = prcr_args->mc_args.N_simulations;

    double square_sum = 0.;
    double final_res = 0.;
    for(size_t i = 0; i < NBLOCKS*TPB;++i){
        
        final_res += results[i].p_off/double(NBLOCKS*TPB);
        square_sum += results[i].p_off2;
    }
    mc_result.value    = final_res;
    mc_result.MC_error = prcr::compute_final_error(square_sum,final_res,NBLOCKS*TPB*PPT);

    return mc_result;
}

__host__ bool
run_device(const prcr::Pricer_args* prcr_args, Result* host_results,const uint * host_seeds)
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
    cudaDeviceSynchronize();


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
        0.,
        prcr_args->schedule_args.T/double(prcr_args->schedule_args.dim),
        prcr_args->schedule_args.dim);

    simulate_device(prcr_args, &starting_point, &schedule, dev_results,dev_seeds);

}


__host__ bool
simulate_host(const prcr::Pricer_args* prcr_args, Result* host_results, const uint * host_seeds)
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
    const prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    Result* results,
    const uint * seeds)
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

    results[index].p_off = pricer.Get_price();
    results[index].p_off2 = pricer.Get_price_square();

}




int main(int argc, char** argv)
{
    using namespace prcr;

    int count;
    cudaGetDeviceCount(&count);
    std::cout << "devices: " << count << std::endl;
    size_t value;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 80000000);
    cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize);
    std::cout << "MallocHeapSize: " << value << std::endl;


    srand(time(NULL));


    std::string filename = "./data/infile_C0_limit_cases.txt";
    std::string outfilename  = "./data/outfile_C0_limit_cases.csv";
    
    Pricer_args* prcr_args = new Pricer_args;
    ReadInputOption(filename, prcr_args);

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    size_t PPT = prcr_args->mc_args.N_simulations;
    
    //gen seeds 
    srand(time(NULL));
    uint* seeds = new uint[4 * NBLOCKS * TPB];


    std::fstream ofs(outfilename.c_str(),std::fstream::out);

        //last_steps
    Result* b0_results = new Result[NBLOCKS * TPB];
    Result* k0_results = new Result[NBLOCKS * TPB];
    Result* k1_results = new Result[NBLOCKS * TPB];
    

    //simulate
    cudaSetDevice(1);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 80000000);
    prcr_args->stc_pr_args.exact = true;
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
        seeds[inc] = rnd::genSeed(true); 
    
    prcr_args->contract_args.B = 0.;
    run_device(prcr_args, b0_results,seeds);
    
    prcr_args->contract_args.B = 0.5;
    prcr_args->contract_args.K = 0;
    run_device(prcr_args, k0_results, seeds);
    
    prcr_args->contract_args.K = 1.;
    run_device(prcr_args, k1_results, seeds);
    
    ofs.close();
}