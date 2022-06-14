#ifndef __PROCESS_EQ__
#define __PROCESS_EQ__

#include "process.cuh"
#include "../equity_lib/equity_prices.cuh"
#include "../support_lib/myRandom/random_numbers.cuh"


// cuda macro
#define H __host__
#define D __device__
#define HD __host__ __device__

class Process_eq : public Process
{
  public:
    HD Process_eq(){};
    HD Process_eq(rnd::MyRandom * gnr)
        :Process(gnr)
    {
    }


    HD virtual Random_numbers * Get_random_structure() = 0;
    HD virtual Equity_prices  * Get_new_prices(Equity_prices * eq_prices_in, 
                                           Random_numbers * w,
                                           double delta_t) = 0;

};

#endif 