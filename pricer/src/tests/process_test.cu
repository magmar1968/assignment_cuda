#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal_multivariante.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_term_structure.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"


//in questo test creiamo path e memorizziamo last step, poi mediamo (no calcolo payoff ecc)

#define STEPS 5  // number of steps
#define NEQ 1  //number of variables
#define NBLOCKS 64  //cuda blocks
#define TPB 64 //threads per block
#define PPT 1 //paths per thread



struct Vanilla_data
{
    char contract_type;
    double strike_price;
};

struct Schedule_data
{
    double tempi[STEPS];
};

struct Eq_description_data
{
    double vol[STEPS];
    char isin_code[12];
    char name[30];
    char currency[20];
    double div_yield;
    double yc[STEPS];
};

struct Eq_prices_data
{
    pricer::udb start_prices[NEQ];
    double start_time;
};





__global__ void kernel(uint*, Schedule_data*, Eq_description_data*, Eq_prices_data*, Vanilla_data*, double*, int*);
D void simulate_device(uint*, Contract_eq_option_vanilla*, double*, int*);
H void simulate_host(uint*, Contract_eq_option_vanilla*, double*, int*);
HD void simulate_generic(uint*, int, Contract_eq_option_vanilla*, double*, int*);




__global__ void kernel(uint* dev_seeds,
    Schedule_data* dev_sched,
    Eq_description_data* dev_descr,
    Eq_prices_data* dev_prices,
    Vanilla_data* dev_vnl_args,
    double* dev_results,
    int* cuda_int_error)
{
    pricer::udb* start_prices = new pricer::udb[NEQ];                           
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = dev_prices->start_prices[i];
    }
    Equity_description** descr = new Equity_description * [NEQ];

    double start_time = dev_prices->start_time;
    Volatility_surface* vol = new Volatility_surface(dev_descr->vol[0]);
    char* currency;
    currency = (dev_descr->currency);
    Yield_curve_term_structure* yc = new Yield_curve_term_structure(currency, dev_descr->yc, dev_sched->tempi, STEPS);

    for (int i = 0; i < NEQ; i++)
    {
        descr[i] = new Equity_description;
        descr[i]->Set_isin_code(dev_descr->isin_code);
        descr[i]->Set_name(dev_descr->name);
        descr[i]->Set_currency(dev_descr->currency);
        descr[i]->Set_dividend_yield(dev_descr->div_yield);
        descr[i]->Set_yc(yc);
        descr[i]->Set_vol_surface(vol);
    }

    Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);
    Schedule* calen = new Schedule(dev_sched->tempi, STEPS);

    Contract_eq_option_vanilla* contr_opt = new Contract_eq_option_vanilla(starting_point_in, calen,
        dev_vnl_args->strike_price, dev_vnl_args->contract_type);
    simulate_device(dev_seeds, contr_opt, dev_results, cuda_int_error);
    for (int i = 0; i < NEQ; i++)
    {
        delete(descr[i]);
    }
    //delete[] (start_prices);
    delete[] (descr);
    delete(vol);
    delete(yc);
    delete(starting_point_in);
    delete(calen);
    delete(contr_opt);



}

D void simulate_device(uint* seeds,
    Contract_eq_option_vanilla* contr_opt,
    double* results,
    int* cuda_int_error)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < NBLOCKS * TPB) { simulate_generic(seeds, index, contr_opt, results, cuda_int_error); }
}

H void simulate_host(uint* seeds,
    Contract_eq_option_vanilla* contr_opt,
    double* results,
    int* host_bool)
{
    for (size_t index = 0; index < NBLOCKS * TPB; index++)
    {
        simulate_generic(seeds, index, contr_opt, results, host_bool);
        
    }
}


HD void simulate_generic(uint* seeds,
    int index,
    Contract_eq_option_vanilla* contr_opt,
    double* results,
    int* status_bool)
{
    uint seed0 = seeds[0 + index * 4];
    uint seed1 = seeds[1 + index * 4];
    uint seed2 = seeds[2 + index * 4];
    uint seed3 = seeds[3 + index * 4];

    rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0, seed1, seed2, seed3);
    Process_eq_lognormal* process = new Process_eq_lognormal(gnr_in);

    Random_numbers* random_numbers_scenario[STEPS];
    Equity_prices* eq_prices_scenario[STEPS];
    
    double delta_t = contr_opt->Get_schedule()->Get_t(0);
    random_numbers_scenario[0] = new Random_numbers(NEQ);
    process->Get_random_structure(random_numbers_scenario[0]);
    Equity_prices* starting_point = contr_opt->Get_eq_prices();
    
    eq_prices_scenario[0] = process->Get_new_prices(starting_point, random_numbers_scenario[0], delta_t);
    
    Schedule* schedule = contr_opt->Get_schedule();
    for (int j = 1; j < STEPS; j++)
    {
    
        delta_t = schedule->Get_t(j) - schedule->Get_t(j - 1);
        random_numbers_scenario[j] = new Random_numbers(NEQ);
        eq_prices_scenario[j] =
            process->Get_new_prices(eq_prices_scenario[j - 1], random_numbers_scenario[j], delta_t);
      
    }
    results[index] =  eq_prices_scenario[STEPS - 1]->Get_eq_price(0).get_number();
    
    delete(gnr_in);
    delete(process);
    for (int i = 0; i < STEPS; i++)
    {
	delete(eq_prices_scenario[i]);
        delete(random_numbers_scenario[i]);
    }
}



int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
    srand(time(NULL));

    int* host_cuda_int_error = new int(0);

    double* host_results = new double[NBLOCKS * TPB];
    for (int i = 0; i < NBLOCKS * TPB; i++)
    {
        host_results[i] = 0;
    }

    uint* seeds = new uint[4 * NBLOCKS * TPB];
    for (size_t inc = 0; inc < 4 * NBLOCKS * TPB; inc++)
    {
        seeds[inc] = rnd::genSeed(true);
    }

    Vanilla_data* vnl_args = new Vanilla_data;

    vnl_args->contract_type = 'C';
    vnl_args->strike_price = 100;

    Schedule_data* sch_args = new Schedule_data;

    double dt = 0.3;
    for (size_t k = 0; k < STEPS; k++)
    {
        sch_args->tempi[k] = (k + 1) * dt;
    }

    Eq_description_data* dscrp_args = new Eq_description_data;

    for (size_t i = 0; i < STEPS; i++)
    {
        dscrp_args->vol[i] = 0.;
    }
    strcpy(dscrp_args->isin_code, "00e99y88o00s");
    strcpy(dscrp_args->name, "opzione di prova");
    strcpy(dscrp_args->currency, "euro");
    dscrp_args->div_yield = 0;
    for (size_t k = 0; k < STEPS; k++)
    {
        dscrp_args->yc[k] = 0.5;
    }



    Eq_prices_data* prices_args = new Eq_prices_data;

    prices_args->start_time = 0;
    for (size_t k = 0; k < NEQ; k++)
    {
        prices_args->start_prices[k] = (k + 1) * 100;
    }




    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;


    if (dev.CPU == true)
    {
        Equity_description** descr = new Equity_description * [NEQ];

        Volatility_surface* vol = new Volatility_surface(dscrp_args->vol[0]);
        char* currency = (dscrp_args->currency);
        Yield_curve_term_structure* yc = new Yield_curve_term_structure(currency, dscrp_args->yc, sch_args->tempi, STEPS);

        for (int i = 0; i < NEQ; i++)
        {
            descr[i] = new Equity_description;
            descr[i]->Set_isin_code(dscrp_args->isin_code);
            descr[i]->Set_name(dscrp_args->name);
            descr[i]->Set_currency(dscrp_args->currency);
            descr[i]->Set_dividend_yield(dscrp_args->div_yield);
            descr[i]->Set_yc(yc);
            descr[i]->Set_vol_surface(vol);
        }

        Equity_prices* starting_point_in = new Equity_prices(prices_args->start_time, &(prices_args->start_prices[0]), NEQ, descr);

        Schedule* calen = new Schedule(sch_args->tempi, STEPS);

        Contract_eq_option_vanilla* contr_opt;
        contr_opt = new Contract_eq_option_vanilla(starting_point_in, calen,
            vnl_args->strike_price, vnl_args->contract_type);
        simulate_host(seeds, contr_opt, host_results, host_cuda_int_error);

        for (int i = 0; i < NEQ; i++)
        {
            delete(descr[i]);
        }
        delete[](descr);
        delete(vol);
        delete(yc);
        //delete(starting_point_in);
        delete(calen);
        delete(contr_opt);

    }


    if (dev.GPU == true)
    {

        //CudaSetDevice(0);
        uint* dev_seeds;
        Schedule_data* dev_sched;
        Eq_description_data* dev_descr;
        Eq_prices_data* dev_prices;
        Vanilla_data* dev_vnl_args;
        double* dev_res = new double[NBLOCKS * TPB];
        int* dev_cuda_int_error;



        cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS * TPB * 4 * sizeof(uint));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_sched, sizeof(Schedule_data));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_descr, sizeof(Eq_description_data));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_prices, sizeof(Eq_prices_data));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc4 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_res, NBLOCKS * TPB * sizeof(double));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc5 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_vnl_args, NBLOCKS * TPB * sizeof(Vanilla_data));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc6 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_cuda_int_error, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc7 failed!\n"); }

        cudaStatus = cudaMemcpy(dev_cuda_int_error, host_cuda_int_error, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy0 failed! %s\n", cudaGetErrorString(cudaStatus)); }

        cudaStatus = cudaMemcpy(dev_seeds, seeds, NBLOCKS * TPB * 4 * sizeof(uint), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_sched, sch_args, sizeof(Schedule_data), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_descr, dscrp_args, sizeof(Eq_description_data), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_prices, prices_args, sizeof(Eq_prices_data), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy4 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_vnl_args, vnl_args, sizeof(Vanilla_data), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy5 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_res, host_results, NBLOCKS * TPB * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy6 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        kernel << < NBLOCKS, TPB >> > (dev_seeds, dev_sched, dev_descr, dev_prices, dev_vnl_args, dev_res, dev_cuda_int_error);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }

        cudaFree(dev_seeds);
        cudaFree(dev_sched);
        cudaFree(dev_descr);
        cudaFree(dev_prices);
        cudaFree(dev_vnl_args);


        cudaStatus = cudaMemcpy(host_results, dev_res, NBLOCKS * TPB * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy backwards1 failed!\n"); }
        printf("%d", cudaStatus);
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(host_cuda_int_error, dev_cuda_int_error, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy back2 failed!\n"); }


        cudaFree(dev_res);
    }

    if (*host_cuda_int_error!=0)
    {
        std::cout << host_cuda_int_error;
        printf("Something went wrong... \n");
    }

    //statistica finale --- scrivere funzione media per struct?
    double final_res;
    for (size_t i = 0; i < NBLOCKS * TPB; i++)
    {
        final_res += host_results[i];
    }
    final_res /= double(PPT * NBLOCKS * TPB);
    //final_res.error = sqrt(final_res.error / double(NBLOCKS * TPB * PPT) - final_res.opt_price * final_res.opt_price);


    delete[](seeds);
    delete(host_cuda_int_error);
    delete[](host_results);
    delete(vnl_args);
    delete(sch_args);
    delete(prices_args);
    delete(dscrp_args);

    std::cout << "\n" << final_res << std::endl;
    //std::cout << final_res.error << std::endl;
    return 0;

    

}
