#include "header.cuh"
#include <iomanip>


#define MAX_M  105
#define MIN_M  9
#define M_STEP 3


void __host__ run_device(const prcr::Pricer_args* prcr_args, const uint *, const uint * );
void __global__ kernel(const prcr::Pricer_args* prcr_args, const  uint *, uint * );
void __host__   simulate_host(prcr::Pricer_args* prcr_args, uint*, uint);
void __device__ simulate_device(const prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, const uint*);
void __host__ __device__ simulate_generic
(size_t, const prcr::Pricer_args*, prcr::Equity_prices*, prcr::Schedule*, const uint*);

__host__ void
run_device(const prcr::Pricer_args* prcr_args,const uint * host_seeds,const uint * host_m)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Pricer_args* dev_prcr_args;
    uint * dev_seeds;
    uint * dev_m;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(Pricer_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_m, sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS * TPB *4 * sizeof(uint));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }




    cudaStatus = cudaMemcpy(dev_prcr_args, prcr_args, sizeof(Pricer_args), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaMemcpy(dev_m, host_m, sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaMemcpy(dev_seeds, host_seeds, NBLOCKS * TPB * 4 * sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }



    kernel << < NBLOCKS, TPB >> > (dev_prcr_args, dev_seeds,dev_m);


    cudaFree(dev_m);
    cudaFree(dev_prcr_args);
    cudaFree(dev_seeds);

    return;
}




__global__ void
kernel(const prcr::Pricer_args* prcr_args, const uint * dev_seeds, uint * dev_m)
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
        prcr_args->schedule_args.T/double(*dev_m),
        *dev_m);

    simulate_device(prcr_args, &starting_point, &schedule,dev_seeds);

}


__host__ void
simulate_host(const prcr::Pricer_args* prcr_args, const uint * host_seeds, const uint * host_m)
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

    Schedule * schedule = new Schedule(
        0.,
        prcr_args->schedule_args.T/double(*host_m),
        *host_m);


    for (int index = 0; index < NBLOCKS * TPB; ++index)
    {
        simulate_generic(index, prcr_args, starting_point, schedule,host_seeds);
    }


    delete(descr);
    delete(starting_point);
    delete(schedule);
    return;
}


__device__ void
simulate_device(
    const prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    const uint * dev_seeds)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = gridDim.x;
    size_t TPB = blockDim.x;
    if (index < NBLOCKS * TPB) simulate_generic(index, prcr_args, starting_point, schedule,dev_seeds);
}

__host__ __device__ void
simulate_generic(size_t index,
    const prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    const uint * seeds)
{

    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];

    double p_off;
    double p_off2;
    rnd::GenCombined gnr_in(seed0,seed1,seed2,seed3);


    prcr::Process_eq_lognormal process(&gnr_in, prcr_args->stc_pr_args.exact);

    prcr::Contract_eq_option_vanilla contr_opt(starting_point,
                                               schedule,
                                               prcr_args->contract_args.strike_price,
                                               prcr_args->contract_args.contract_type);
    size_t _N = prcr_args->mc_args.N_simulations;
    prcr::Option_pricer_montecarlo pricer(&contr_opt, &process, _N);

    p_off = pricer.Get_price();
    p_off2 = pricer.Get_price_square();

}




int main(int argc, char** argv)
{
    using namespace prcr;


    srand(time(NULL));


    std::string filename = "./data/infile_B5b_g.txt";
    std::string outfilename  = "./data/outfile_B5b_g.csv";
    
    Pricer_args* prcr_args = new Pricer_args;
    ReadInputOption(filename, prcr_args);

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;
    
    //gen seeds 
    srand(time(NULL));
    uint* seeds = new uint[4 * NBLOCKS * TPB];
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
        seeds[inc] = rnd::genSeed(true); 



    std::fstream ofs(outfilename.c_str(),std::fstream::out);
    ofs << "m,cpu_time,gpu_time,g\n";

    cudaSetDevice(1);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 80000000);
    uint * m_array = new uint[MAX_M];
    size_t m_cont = 0;
    
    for (size_t m = MIN_M; m < MAX_M; m+=M_STEP){
        m_array[m_cont] = m;
        //simulate
        double time_gpu= 0,time_cpu=0;
        Timer timer_gpu;
        run_device(prcr_args,seeds,&m_array[m_cont]);
        time_gpu = timer_gpu.GetDeltamsTime();
        
        Timer timer_cpu;
        simulate_host(prcr_args,seeds,&m_array[m_cont]);
        time_cpu = timer_cpu.GetDeltamsTime();
         
        
        double g = time_cpu/time_gpu;
        

        ofs << m
            << "," << time_cpu
            << "," << time_gpu 
            << "," << g << "\n";
       
        std::cout << "currently at:" << double(m)/double(MAX_M) * 100 << "% " << std::endl;
        m_cont ++;
    }
    delete[](m_array);
    ofs.close();
}