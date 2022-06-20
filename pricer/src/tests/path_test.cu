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


#define NPATH 5  //number of paths
#define STEPS 5  // number of steps
#define NEQ 5  //number of equities


__global__ void kernel(double*);
D void createPath_device(Process_eq*, Equity_prices*,  Schedule*, Path*, size_t, double*);
H void createPath_host(Process_eq*, Equity_prices*, Schedule*, Path*, size_t);
HD void createPath_generic(Process_eq*, Equity_prices*, Schedule*, Path* , size_t);




__global__ void kernel(double* path_out)
{
    rnd::GenCombined* gnr_in = new rnd::GenCombined(800, 200, 400, 500);
    Process_eq_lognormal_multivariante* process_in = new Process_eq_lognormal_multivariante(gnr_in, NEQ);
    pricer::udb start_prices[NEQ];
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = 100 * (1 + i) + i;
    }
    double start_time = 0.15;
    Equity_description** descr = new Equity_description * [NEQ];

    Volatility_surface* vol = new Volatility_surface(0.01);
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

    Path* cammini_GPU = new Path[NPATH];

    createPath_device(process_in, starting_point_in, calen, cammini_GPU, NPATH, path_out);
}

D void createPath_device(Process_eq* process,
    Equity_prices* starting_point,
    Schedule* calendar,
    Path* cammini,
    size_t totpaths,
    double* path_out)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < totpaths)
    {
        createPath_generic(process, starting_point, calendar, cammini, index);
	
	for (size_t j = 0; j < NEQ; j++)
        {
            //path_out[i] 0 = new double[STEPS];
            for (size_t k = 0; k < STEPS; k++)
            {
                path_out[index] = cammini[index].Get_equity_prices(0)->Get_eq_price(0).get_number();
            }
        }
        index += blockDim.x * gridDim.x;
    }
}

H void createPath_host(Process_eq* process,
    Equity_prices* starting_point,
    Schedule* calendar,
    Path* cammini,
    size_t totpaths)
{
    for (size_t index = 0; index < totpaths; index++)
    {
        createPath_generic(process, starting_point, calendar, cammini, index);
    }
}


HD void createPath_generic(Process_eq* process,
    Equity_prices* starting_point,
    Schedule* calendar,
    Path* cammini,
    size_t index)
{
        cammini[index] = Path(starting_point, calendar, process);
}


int main(int argc, char **argv)
{

    srand(time(NULL));
    cudaError_t cudaStatus;

    size_t* npath = new size_t(NPATH);
    size_t* neq = new size_t(NEQ);

    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;

    if (dev.CPU == true)
    {

        rnd::GenCombined* gnr_in = new rnd::GenCombined(800, 200, 400, 500);
        Process_eq_lognormal_multivariante* process_in = new Process_eq_lognormal_multivariante(gnr_in, NEQ);
        pricer::udb start_prices[NEQ];
        for (int i = 0; i < NEQ; i++)
        {
            start_prices[i] = 100 * (1 + i) + i;
        }
        double start_time = 0.15;
        Equity_description** descr = new Equity_description * [NEQ];

        Volatility_surface* vol = new Volatility_surface(0.01);
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




        Path* cammini_CPU = new Path[NPATH];
        createPath_host(process_in, starting_point_in, calen, cammini_CPU, NPATH);


        //stampa CPU

        std::cout << std::endl << "paths:" << std::endl;
        for (int i = 0; i < NPATH; i++)
        {
            std::cout << "\n\n\n\npath " << i << ":" << std::endl;
            for (int k = 0; k < NEQ; k++)
            {
                std::cout << "\nequity " << k << ":" << std::endl;
                for (int j = 0; j < STEPS; j++)
                    std::cout << cammini_CPU[i].Get_equity_prices(j)->Get_eq_price(k).get_number() << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }










    if (dev.GPU == true)
    {
        //CudaSetDevice(0);
        double* dev_paths;
        double* paths = new double[NPATH];
	/*for (size_t t = 0; t < NPATH; t++)
	{
		paths[t] = new double*[STEPS];
		for(size_t y = 0; y < STEPS; y++)
		{
			paths[t][y] = new double[NEQ];
		}
	}*/ 	

        cudaStatus = cudaMalloc((void**)&dev_paths, NPATH*sizeof(double));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!\n"); }

        kernel << <32, 32 >> > (dev_paths);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }

        cudaStatus = cudaMemcpy(paths,dev_paths, NPATH * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!\n"); }
	fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
        //stampa  GPU

        std::cout << std::endl << "paths:" << std::endl;
        for (int i = 0; i < NPATH; i++)
        {
            std::cout << "\n\n\n\npath " << i << ":" << std::endl;
            for (int k = 0; k < NEQ; k++)
            {
                std::cout << "\nequity " << k << ":" << std::endl;
                for (int j = 0; j < STEPS; j++)
                    std::cout << paths[i];
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }

    }
    


    return 0;

}


/*cudaStatus = cudaMalloc((void**)&dev_process, sizeof(Process_eq));
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc 1 failed!\n"); }

cudaStatus = cudaMalloc((void**)&dev_prices, sizeof(Equity_prices));
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc 2 failed!\n"); }

cudaStatus = cudaMalloc((void**)&dev_schedule, sizeof(Schedule));
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc 3 failed!\n"); }

cudaStatus = cudaMalloc((void**)&dev_totpaths, sizeof(size_t));
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc 4 failed!\n"); }

cudaStatus = cudaMalloc((void**)&dev_paths, NPATH * sizeof(Path));
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc 6 failed!\n"); }




cudaStatus = cudaMemcpy(dev_process, process_in, sizeof(Process_eq), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy 1 failed!\n"); }

cudaStatus = cudaMemcpy(dev_prices, starting_point_in, sizeof(Equity_prices), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy 2 failed!\n"); }

cudaStatus = cudaMemcpy(dev_schedule, calen, sizeof(Schedule), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy 3 failed!\n"); }

cudaStatus = cudaMemcpy(dev_totpaths, npath, sizeof(size_t), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy 4 failed!\n"); }



kernel << <32, 32 >> > (process_in, starting_point_in, calen, cammini_GPU, NPATH);
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }


cudaStatus = cudaMemcpy(cammini_GPU, dev_paths, NPATH * sizeof(Path), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy backward failed!\n"); }*/
