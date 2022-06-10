#ifndef __PROCESS_EQ__
#define __PROCESS_EQ__

#include "process.cuh"
#include "../equity_lib/equity_prices.cuh"

class Process_eq : public Process
{
  public:
    Process_eq(){};
    Process_eq(rnd::MyRandom * gnr)
        :Process(gnr)
    {
    }

    virtual ~Process_eq(){};

    virtual double * Get_random_strucure() = 0;
    virtual Equity_prices * Get_new_prices(Equity_prices * eq_prices_in, 
                                           double *w,
                                           double delta_t) = 0;

};

#endif 