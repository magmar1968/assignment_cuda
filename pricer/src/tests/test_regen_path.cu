#include "header.cuh"
#include <iomanip>

#define NSIM 1

bool __host__ run_device        (prcr::Pricer_args* prcr_args, double* host_last_steps);
void __global__ kernel          (prcr::Pricer_args* prcr_args, double* dev_last_steps);
bool __host__   simulate_host   (prcr::Pricer_args* prcr_args, double* host_last_steps);
void __device__ simulate_device (prcr::Pricer_args* prcr_args, prcr::Equity_prices*, prcr::Schedule*, double* dev_last_steps);
void __host__ __device__ simulate_generic
                                (size_t, prcr::Pricer_args* , prcr::Equity_prices*, prcr::Schedule*, double*);

__host__ bool
run_device(prcr::Pricer_args* prcr_args, double* host_last_steps)
{
    using namespace prcr;
    cudaError_t cudaStatus;
    double* dev_last_steps;
    Pricer_args* dev_prcr_args;

    size_t NBLOCKS = prcr_args->dev_opts.N_blocks;
    size_t TPB = prcr_args->dev_opts.N_threads;

    cudaStatus = cudaMalloc((void**)&dev_prcr_args, sizeof(Pricer_args));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

    cudaStatus = cudaMalloc((void**)&dev_last_steps, NBLOCKS * TPB * sizeof(double));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }



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

    kernel << < NBLOCKS, TPB >> > (dev_prcr_args, dev_last_steps);

    cudaStatus = cudaMemcpy(host_last_steps, dev_last_steps, NBLOCKS * TPB * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){ 
        fprintf(stderr, "cudaMemcpy3 failed!\n"); 
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_last_steps);
    cudaFree(dev_prcr_args);

    return cudaStatus;
}




__global__ void
kernel(prcr::Pricer_args* prcr_args, double* dev_last_steps)
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



    simulate_device( prcr_args, starting_point, schedule, dev_last_steps);

   
    delete(descr);
    delete(starting_point);
    delete(schedule);
}


__host__ bool
simulate_host(prcr::Pricer_args* prcr_args, double* last_steps)
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
        simulate_generic(index, prcr_args, starting_point, schedule, last_steps);
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
    double* last_steps)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t NBLOCKS = gridDim.x;
    size_t TPB = blockDim.x;
    if (index < NBLOCKS * TPB) simulate_generic(index, prcr_args, starting_point, schedule, last_steps);
}

__host__ __device__ void
simulate_generic(size_t index,
    prcr::Pricer_args* prcr_args,
    prcr::Equity_prices* starting_point,
    prcr::Schedule* schedule,
    double* last_steps)
{

    rnd::MyRandomDummy* gnr_in = new rnd::MyRandomDummy();
    prcr::Process_eq_lognormal* process 
                = new prcr::Process_eq_lognormal(gnr_in,prcr_args->stc_pr_args.exact);

                
    prcr::Path path(starting_point, schedule, process);
    last_steps[index] = path.Get_last_eq_price();
    for(int i = 0; i < prcr_args->mc_args.N_simulations; ++i){
        path.regen_path();
        if(last_steps[index] == path.Get_last_eq_price()){
            continue;
        }else
            last_steps[index] = -1000;
    }

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
        exact_value = 108.223272744281;
    }else
        exact_value = 108.306658813969;

    bool GPU = prcr_args->dev_opts.GPU;
    bool CPU = prcr_args->dev_opts.CPU;
    bool status = true;
    
    if (GPU == true) 
    {
        for(int i = 0; i < N_TEST_SIM; ++i)
            status = status && run_device(prcr_args, last_steps_gpu);
        for (int j = 0; j < NBLOCKS * TPB; j++)
        {   
            double delta = abs(last_steps_gpu[j] - exact_value);
            if( delta > std::pow(10,-12)){
                std::cerr << "ERROR: thread " << j << " failed to regen path. "
                          << "Value: " << last_steps_gpu[j] << std::endl;
                last_step_check_gpu = false;
            }
        }
	std::cout << last_steps_gpu[NBLOCKS*TPB-1] << std::endl;
		
    }
    

    if (CPU == true) 
    {
        for(int i = 0; i < N_TEST_SIM; ++i)
            status = status && simulate_host(prcr_args, last_steps_cpu);
        for (int j = 0; j < NBLOCKS * TPB; j++)
        {
            double delta = abs(last_steps_cpu[j] - exact_value);
            if(delta > std::pow(10,-12)){
                std::cerr << "ERROR: thread " << j << " failed to regen path. "
                          << "Value: " << last_steps_cpu[j] << std::endl;
                last_step_check_cpu = false;
            }
        }
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
