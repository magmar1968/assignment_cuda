// #include <iostream>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// //#include "../lib/pathgen_lib/path.cu"
// #include "../lib/pathgen_lib/path.cuh"
// //#include "../lib/support_lib/myRandom/myRandom.cu"
// #include "../lib/support_lib/myRandom/myRandom.cuh"
// #include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
// //#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cu"
// #include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
// //#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cu"
// #include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
// //#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cu"
// //#include "../lib/pathgen_lib/process_eq_imp/process_eq_lognormal_multivariante.cu"
// #include "../lib/pathgen_lib/process_eq_imp/process_eq_lognormal_multivariante.cuh"
// //#include "../lib/pathgen_lib/process_eq_imp/process_eq_lognormal.cu"
// #include "../lib/pathgen_lib/process_eq_imp/process_eq_lognormal.cuh"
// #include "../lib/equity_lib/schedule_lib/schedule.cuh"
// //#include "../lib/equity_lib/schedule_lib/schedule.cu"
// #include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
// #include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"





// //interfaccia per gpu cpu


// //__global__ void kernel();
// //__device__ void createPath_device();
// H void createPath_host(Process_eq*, Equity_prices*, Equity_prices**, Schedule*, Path*, size_t);     //H
// HD void createPath_generic(Process_eq*, Equity_prices*, Equity_prices**, Schedule*, Path*, size_t, size_t); //HD




// /*__global__ void kernel(double** paths,
//     uint* seeds,
//     pricer::Schedule* cal,
//     int       dim,
//     size_t    path_len)
// {
//     createPath_device(paths, seeds, cal, dim, path_len);
// }*/

// /*__device__ void createPath_device(double** paths,
//     uint* seeds,
//     pricer::Schedule* cal,
//     int       dim,
//     size_t    path_len)
// {
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//     createPath_generic(paths, seeds, index, cal, dim, path_len);
// }*/

// H void createPath_host(Process_eq* process,
//     Equity_prices* starting_point,
//     Equity_prices** eq_prices_scenario,
//     Schedule* calendar,
//     Path* cammini,
//     size_t totpaths)
// {
//     for (size_t index = 0; index < totpaths; index++)
//     {
//         size_t dim = cammini[0].Get_dim();
//         createPath_generic(process, starting_point, eq_prices_scenario, calendar, cammini, index, dim);
//     }
// }


// HD void createPath_generic(Process_eq* process,
//     Equity_prices* starting_point,
//     Equity_prices** eq_prices_scenario,
//     Schedule* calendar,
//     Path* cammini,
//     size_t index,
//     size_t dim)
// {
//     if (index < dim)
//     {
//         cammini[index] = Path(starting_point, calendar, process);
//     }
// }




// #define NPATH 3   //number of paths
// #define STEPS 5   // number of steps
// #define NEQ 4     //number of equities

// int main()
// {

//     srand(time(NULL));
//     cudaError_t cudaStatus;


//     rnd::GenCombined* gnr_in = new rnd::GenCombined(800, 200, 400, 500);
//     Process_eq_lognormal_multivariante* process_in = new Process_eq_lognormal_multivariante(gnr_in, NEQ);
//     double start_prices[4] = { 200, 50, 100, 1000 };
//     double start_time = 0.3;
//     Equity_description** descr = new Equity_description * [NEQ];

//     Volatility_surface* vol = new Volatility_surface(1.5);
//     Yield_curve_flat* yc = new Yield_curve_flat("euro", 2.0);

//     for (int i = 0; i < NEQ; i++)
//     {
//         descr[i] = new Equity_description;
//         descr[i]->Set_isin_code("isin codein");
//         descr[i]->Set_name("namein ");
//         descr[i]->Set_currency("currencyin");
//         descr[i]->Set_dividend_yield(1.1);
//         descr[i]->Set_yc(yc);
//         descr[i]->Set_vol_surface(vol);
//     }

//     Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);
//     Equity_prices** eq_prices_scenario_in = new Equity_prices * [NEQ];
//     for (int i = 0; i < NEQ; i++)
//     {
//         eq_prices_scenario_in[i] = new Equity_prices[STEPS];
//     }
//     double tempi[STEPS] = { 0.2, 0.4, 0.6, 0.8, 1 };
//     Schedule* calen = new Schedule(tempi, 5);
//     Path* cammini = new Path[NPATH];

//     /*CudaSetDevice(0);
//      cudaStatus = cudaMalloc((void**)&dev_seeds, NPATH * sizeof(int));
//      if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!\n"); }

//      cudaMalloc((void**)&dev_paths, NPATH * STEPS * sizeof(double));
//      cudaMemcpy(dev_seeds, seeds, NPATH, cudaMemcpyHostToDevice);

//      kernel << <NPATH, 1 >> > (dev_paths, dev_seeds, NPATH, STEPS);

//      cudaMemcpy(paths, dev_paths, NPATH * STEPS * sizeof(double), cudaMemcpyDeviceToHost);*/

//     createPath_host(process_in, starting_point_in, eq_prices_scenario_in, calen, cammini, NPATH);

//     //statistica

//     std::cout << "createpathhost passato\n";
//     std::cout << std::endl << "paths:" << std::endl;
//     for (int i = 0; i < NPATH; i++)
//     {
//         std::cout << "path " << i << ":" << std::endl;
//         for (int k = 0; k < NEQ; k++)
//         {
//             std::cout << "equity" << k << ":" << std::endl;
//             for (int j = 0; j < STEPS; j++)
//                 std::cout << cammini[i].Get_equity_prices(j)->Get_eq_price(k) << " ";
//         }
//         std::cout << std::endl;
//         std::cout << std::endl;
//     }


//     return 0;

// }


