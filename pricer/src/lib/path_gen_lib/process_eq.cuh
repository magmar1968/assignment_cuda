#ifndef __PROCESS_EQ__
#define __PROCESS_EQ__

#include "process.cuh"
#include "../equity_lib/equity_prices.cuh"
#include "../../support_lib/myRandom/random_numbers.cuh"

class Process_eq : public Process
{
  public:
    Process_eq(){};
    Process_eq(rnd::MyRandom * gnr)
        :Process(gnr)
    {
    }


    virtual Random_numbers * Get_random_strucure() = 0;
    virtual Equity_prices  * Get_new_prices(Equity_prices * eq_prices_in, 
                                           Random_numbers * w,
                                           double delta_t) = 0;

};

#endif 