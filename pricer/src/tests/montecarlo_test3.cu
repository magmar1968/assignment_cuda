#include "header.cuh"
#include <iomanip>

#define NSIM 1


struct Result
{
    double p_off;
    double p_off2;
};

bool __host__ run_device(prcr::Pricer_args* prcr_args, Result* host_results);
void __global__ kernel(prcr::Pricer_args* prcr_args, Result* dev_results);
bool __host__   simulate_host(prcr::Pricer_args* prcr_args, Result* host_results);
void __device__ simulate_device(prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, Result* dev_results);
void __host__ __device__ simulate_generic
(size_t, prcr::Pricer_args*, prcr::Equity_prices*, prcr::Schedule*, Result*);

__host__ bool
run_device(prcr::Pricer_args* prcr_args, Result* host_results)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    Result* dev_results;
    Pricer_args* dev_prcr_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(Pricer_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_results, NBLOCKS * TPB * sizeof(Result));
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

    kernel << < NBLOCKS, TPB >> > (dev_prcr_args, dev_results);

    cudaStatus = cudaMemcpy(host_results, dev_results, NBLOCKS * TPB * sizeof(Result), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_results);
    cudaFree(dev_prcr_args);

    return cudaStatus;
}




__global__ void
kernel(prcr::Pricer_args* prcr_args, Result* dev_results)
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



    simulate_device(prcr_args, starting_point, schedule, dev_results);


    delete(descr);
    delete(starting_point);
    delete(schedule);
}


__host__ bool
simulate_host(prcr::Pricer_args* prcr_args, Result* host_results)
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
        simulate_generic(index, prcr_args, starting_point, schedule, host_results);
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
    Result* dev_results)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = gridDim.x;
    size_t TPB = blockDim.x;
    if (index < NBLOCKS * TPB) simulate_generic(index, prcr_args, starting_point, schedule, dev_results);
}

__host__ __device__ void
simulate_generic(size_t index,
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    Result* results)
{

    rnd::MyRandomDummy* gnr_in = new rnd::MyRandomDummy();
    prcr::Process_eq_lognormal* process
        = new prcr::Process_eq_lognormal(gnr_in, prcr_args->stc_pr_args.exact);

    prcr::Contract_eq_option_vanilla contr_opt(starting_point,
                                               schedule,
                                               prcr_args->contract_args.strike_price,
                                               prcr_args->contract_args.contract_type);
    double pay_off = 0.;
    double pay_off2 = 0.;


    prcr::Contract_eq_option& contract =
        static_cast<prcr::Contract_eq_option&>(contr_opt);
    prcr::Path path(starting_point, schedule, process);
    size_t _N = prcr_args->mc_args.N_simulations;
    for (size_t i = 0; i < _N; ++i)
    {

        pay_off += contract.Pay_off(&path);
        pay_off2 += contract.Pay_off(&path) * contract.Pay_off(&path);

        path.regen_path();
    }

    results[index].p_off = pay_off / double(_N);
    results[index].p_off2 = pay_off2;


    delete(process);
    delete(gnr_in);
}




int main(int argc, char** argv)
{
    using namespace prcr;


    srand(time(NULL));


    std::string filename = "./data/infile_test_regen_path.txt";
    Pricer_args* prcr_args = new Pricer_args;
    ReadInputOption(filename, prcr_args);

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

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

    if (prcr_args->stc_pr_args.exact == true) {
        exact_value = 8.223272744281;
    }
    else
        exact_value = 8.306658813969;

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;

    if (GPU == true)
    {
        for (int i = 0; i < N_TEST_SIM; ++i)
            status = status && run_device(prcr_args, gpu_results);
        for (int j = 0; j < NBLOCKS * TPB; j++)
        {
            double delta = abs(gpu_results[j].p_off - exact_value);
            if (delta > std::pow(10, -12)) {
                std::cerr << "ERROR: thread " << j << " failed to regen path. "
                    << "Value: " << gpu_results[j].p_off << std::endl;
                results_check_gpu = false;
            }
        }
        std::cout << gpu_results[NBLOCKS * TPB - 1].p_off << std::endl;

    }


    if (CPU == true)
    {
        for (int i = 0; i < N_TEST_SIM; ++i)
            status = status && simulate_host(prcr_args, cpu_results);
        for (int j = 0; j < NBLOCKS * TPB; j++)
        {
            double delta = abs(cpu_results[j].p_off - exact_value);
            if (delta > std::pow(10, -12)) {
                std::cerr << "ERROR: thread " << j << " failed to regen path. "
                    << "Value: " << cpu_results[j].p_off << std::endl;
                results_check_cpu = false;
            }
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
