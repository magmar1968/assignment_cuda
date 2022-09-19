#include <iostream>
#include "cuda_runtime.h"
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
#include "../lib/contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"


//in questo test creiamo path e memorizziamo last step, poi mediamo (no calcolo payoff ecc)

#define STEPS 5  // number of steps
#define NEQ 1  //number of equities
#define NBLOCKS 16  //cuda blocks
#define TPB 512 //threads per block
#define PPT 1



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





__global__ void kernel_mc(uint*, Schedule_data*, Eq_description_data*, Eq_prices_data*, Vanilla_data*, double*, int*);
D void simulate_device(uint*, Contract_eq_option_vanilla*, double*, int*);
H void simulate_host(uint*, Contract_eq_option_vanilla*, double*, int*);
HD void simulate_generic(uint*, int, Contract_eq_option_vanilla*, double*, int*);




__global__ void kernel_mc(uint* dev_seeds,
    Schedule_data* dev_sched,
    Eq_description_data* dev_descr,
    Eq_prices_data* dev_prices,
    Vanilla_data* dev_vnl_args,
    double* dev_results,
    int* cuda_bool)
{
    pricer::udb* start_prices = new pricer::udb[NEQ];                              //definiamo oggetti dentro a kernel
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = dev_prices->start_prices[i];
    }
    Equity_description** descr= new Equity_description*[NEQ];

    double start_time = dev_prices->start_time;
    Volatility_surface* vol =new Volatility_surface(dev_descr->vol[0]);
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

    Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, descr);
    Schedule* calen = new Schedule(dev_sched->tempi, STEPS);

    Contract_eq_option_vanilla* contr_opt = new Contract_eq_option_vanilla(starting_point_in, calen,
        dev_vnl_args->strike_price, dev_vnl_args->contract_type);
    simulate_device(dev_seeds, contr_opt, dev_results, cuda_bool);
    for (size_t i = 0; i < NEQ; i++)
    {
        delete(descr[i]);
    }
    //delete[] (start_prices);
    //delete[] (descr);
    /*delete(vol);
    delete(yc);
    delete(starting_point_in);
    delete(calen);
    delete(contr_opt);*/



}

D void simulate_device(uint* seeds,
    Contract_eq_option_vanilla* contr_opt,
    double* results,
    int* cuda_bool)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    /*if (index < NBLOCKS * TPB)*/ simulate_generic(seeds, index, contr_opt, results, cuda_bool);
}

H void simulate_host(uint* seeds,
    Contract_eq_option_vanilla* contr_opt,
    double* results,
    int* host_bool)
{
    for (size_t index = 0; index < NBLOCKS * TPB; index++)   //l'espressione che risulta dalla moltiplicazione � il numero totale di threads
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

    rnd::MyRandomDummy* gnr_in = new rnd::MyRandomDummy();//(seed0, seed1, seed2, seed3);
    Process_eq_lognormal_multivariante* process = new Process_eq_lognormal_multivariante(gnr_in, NEQ);
    
    Path* cammino = new Path(contr_opt->Get_eq_prices(), contr_opt->Get_schedule(), &static_cast<Process_eq&>(*process));
    results[index] += cammino->Get_equity_prices(STEPS - 1)->Get_eq_price(0).get_number();
    for (int i = 1; i < PPT; i++)
    {
        cammino->regen_path(contr_opt->Get_schedule(), &static_cast<Process_eq&>(*process));
        results[index] += cammino->Get_equity_prices(STEPS - 1)->Get_eq_price(0).get_number();
	//cammino->destroy();
    }
    delete(cammino);
    delete(gnr_in);
    delete(process);
}



int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
    srand(time(NULL));

    int* host_cuda_bool = new int(0);
    //*host_cuda_bool = true;

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
        dscrp_args->vol[i] = 0.1;
    }
    strcpy(dscrp_args->isin_code, "qwertyuiopas");
    strcpy(dscrp_args->name, "opzione di prova");
    strcpy(dscrp_args->currency, "euro");
    dscrp_args->div_yield = 0;
    for (size_t k = 0; k < STEPS; k++)
    {
        dscrp_args->yc[k] = 0.1;
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

        simulate_host(seeds, contr_opt, host_results, host_cuda_bool);
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
        int* dev_cuda_bool = new int(0);



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
       
        cudaStatus = cudaMalloc((void**)&dev_cuda_bool, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc7 failed!\n"); }

        cudaStatus = cudaMemcpy(dev_cuda_bool, host_cuda_bool, sizeof(int), cudaMemcpyHostToDevice);
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

        cudaStatus = cudaMemcpy(dev_res,host_results, NBLOCKS*TPB*sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy6 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        kernel_mc << < NBLOCKS, TPB >> > (dev_seeds, dev_sched, dev_descr, dev_prices, dev_vnl_args, dev_res, dev_cuda_bool);
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

        cudaStatus = cudaMemcpy(host_cuda_bool, dev_cuda_bool, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy back2 failed!\n"); }











    }

    /*if (!*host_cuda_bool)
    {
        printf("Something went wrong... \n"); //codice di errore intero?
    }*/

    //statistica finale --- scrivere funzione media per struct?
    double final_res;
    for (size_t i = 0; i < NBLOCKS * TPB; i++)
    {
        final_res += host_results[i];
    }
    final_res /= double(PPT*NBLOCKS * TPB);
    //final_res.error = sqrt(final_res.error / double(NBLOCKS * TPB * PPT) - final_res.opt_price * final_res.opt_price);

    std::cout << "\n" << final_res << std::endl;
    //std::cout << final_res.error << std::endl;
    return 0;

}
