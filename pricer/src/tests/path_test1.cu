#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal_multivariante.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"


#define NPATH 1  //number of paths
#define STEPS 5  // number of steps
#define NEQ 5  //number of equities


__global__ void kernel(double*, uint* );
D void createPath_device(Equity_prices*, Schedule*, double*, size_t, uint*);
H void createPath_host(Equity_prices*, Schedule*, double*, size_t, uint*);
HD void createPath_generic(Process_eq*, Equity_prices*, Schedule*, double* , size_t);




__global__ void kernel(double* path_out, uint* seeds)
{
    pricer::udb start_prices[NEQ];
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = 100 * i;
    }
    double start_time = 0.15;
    Equity_description** descr = new Equity_description * [NEQ];

    Volatility_surface* vol = new Volatility_surface(0.5);
    Yield_curve_flat* yc = new Yield_curve_flat("euro", 0);

    for (int i = 0; i < NEQ; i++)
    {
        descr[i] = new Equity_description;
        descr[i]->Set_isin_code("isin codein");
        descr[i]->Set_name("namein ");
        descr[i]->Set_currency("currencyin");
        descr[i]->Set_dividend_yield(0);
        descr[i]->Set_yc(yc);
        descr[i]->Set_vol_surface(vol);
    }

    Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);
    double tempi[STEPS];
    for (size_t k = 0; k < STEPS; k++)
    {
        tempi[k] = 0.2 + k * 0.2;
    }
    Schedule* calen = new Schedule(tempi, STEPS);

    for (size_t i = 0; i < NEQ; i++)
    {
        path_out[i] = 0;
    }
    createPath_device(starting_point_in, calen, path_out, NPATH, seeds);
    __syncthreads();
   
}

D void createPath_device(Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out,
    size_t totpaths,
    uint* seeds)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint seed0;
    uint seed1;
    uint seed2;
    uint seed3;
    //while (index < totpaths)
        
    //{
        seed0 = seeds[0 + index*4];
        seed1 = seeds[1 + index*4];
        seed2 = seeds[2 + index*4];
        seed3 = seeds[3 + index*4];

        rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0, seed1, seed2, seed3);
        Process_eq_lognormal_multivariante* process = new Process_eq_lognormal_multivariante(gnr_in, NEQ);

        createPath_generic(process, starting_point, calendar, path_out, index);
      

       // index += blockDim.x * gridDim.x;
    //}
}

H void createPath_host(Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out,
    size_t totpaths,
    uint* seeds)
{
    uint seed0;
    uint seed1;
    uint seed2;
    uint seed3;
    for (size_t index = 0; index < totpaths; index++)
    {
        seed0 = seeds[0 + index*4];
        seed1 = seeds[1 + index*4];
        seed2 = seeds[2 + index*4];
        seed3 = seeds[3 + index*4];
        rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0, seed1, seed2, seed3);
        Process_eq_lognormal_multivariante* process = new Process_eq_lognormal_multivariante(gnr_in, NEQ);

        createPath_generic(process, starting_point, calendar, path_out, index);
    }
}


HD void createPath_generic(Process_eq* process,
    Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out ,
    size_t index)
{
        Path cammino = Path(starting_point, calendar, process);
        for (size_t i = 0; i < NEQ; i++)
        {
            path_out[i] = cammino.Get_equity_prices(STEPS - 1)->Get_eq_price(i).get_number();
        }
}
        


int main(int argc, char **argv)
{
    cudaError_t cudaStatus;
    srand(time(NULL));
    size_t* npath = new size_t(NPATH);
    size_t* neq = new size_t(NEQ);

    uint* seeds = new uint[4 * NPATH];
    for (size_t inc = 0; inc < 4 * NPATH; inc++)
    {
        seeds[inc] = rnd::genSeed(true);       //same seeds for gpu and cpu
    }

    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    //if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;

        if (dev.CPU == true)
        {

            pricer::udb start_prices[NEQ];
            for (int i = 0; i < NEQ; i++)
            {
                start_prices[i] = 100 * i;
            }
            double start_time = 0.15;
            Equity_description** descr = new Equity_description * [NEQ];

            Volatility_surface* vol = new Volatility_surface(0.5);
            Yield_curve_flat* yc = new Yield_curve_flat("euro", 0);

            for (int i = 0; i < NEQ; i++)
            {
                descr[i] = new Equity_description;
                descr[i]->Set_isin_code("isin codein");
                descr[i]->Set_name("namein ");
                descr[i]->Set_currency("currencyin");
                descr[i]->Set_dividend_yield(0);
                descr[i]->Set_yc(yc);
                descr[i]->Set_vol_surface(vol);
            }

            Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);

            double tempi[STEPS];
            for (size_t k = 0; k < STEPS; k++)
            {
                tempi[k] = 0.2 + k * 0.2;
            }
            Schedule* calen = new Schedule(tempi, STEPS);

            double* path_CPU = new double[NEQ];


            createPath_host(starting_point_in, calen, path_CPU, NPATH, seeds);


            //stampa CPU
            std::cout << std::endl << "results CPU:\n\n" << std::endl;
            for (int k = 0; k < NEQ; k++)
            {
                std::cout << "\nequity " << k << ":";
                std::cout << "\t" << path_CPU[k] << "\n";
            }
            std::cout << std::endl;
            if (dev.GPU == true)
            {

                //CudaSetDevice(0);
                double* dev_path;
                double* path_GPU = new double[NEQ];
                /*for (size_t t = 0; t < NPATH; t++)
                {
                    paths[t] = new double*[STEPS];
                    for(size_t y = 0; y < STEPS; y++)
                    {
                        paths[t][y] = new double[NEQ];
                    }
                }*/

                uint* dev_seeds;

                cudaStatus = cudaMalloc((void**)&dev_path, NEQ * sizeof(double));
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

                cudaStatus = cudaMalloc((void**)&dev_seeds, NPATH * 4 * sizeof(uint));
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

                cudaStatus = cudaMemcpy(dev_seeds, seeds, NPATH * 4 * sizeof(uint), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }
                fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

                kernel << <1, 1 >> > (dev_path, dev_seeds);
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }

                cudaStatus = cudaMemcpy(path_GPU, dev_path, NEQ * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
                fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
                //stampa  GPU
		std::cout << "results GPU:\n";
                for (int k = 0; k < NEQ; k++)
                {
                    std::cout << "\nequity " << k << ":";
                    std::cout << "\t" << path_GPU[k] << "\n";
                }






                //confronto con cpu
                int err_msg = 0;
		double minimum;
                for (int k = 0; k < NEQ; k++)
                {
		    minimum = min(path_CPU[k], path_GPU[k]);   
                    if (abs(path_CPU[k] - path_GPU[k])>0.000001*minimum) err_msg += pow(2, k) ;
                }
		if(err_msg == 0) {std::cout << "path_test1 ok!\n";}
                return err_msg;
            }
        }
    std::cout << std::endl;
    return 0;
}
