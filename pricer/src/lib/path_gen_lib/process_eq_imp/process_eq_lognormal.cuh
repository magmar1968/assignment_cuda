#ifndef __PROCESS_EQ_LOGNORMAL__
#define __PROCESS_EQ_LOGNORMAL__

#include "../process.cuh"
#include "../../support_lib/myDouble_lib/myudouble.cuh"
#include "../../equity_lib/equity_prices.cuh"

// cuda macro
#define H __host__
#define D __device__
#define HD __host__ __device__ 


class Process_eq_lognormal : public Process
{
  public:
    HD Process_eq_lognormal(){};
    HD Process_eq_lognormal(rnd::MyRandom * gnr);
    HD virtual ~Process_eq_lognormal(){};
    //functions
    HD pricer::udb Get_new_equity_price(
              Equity_description * eq_descr,
              pricer::udb eq_price,
              double w, //random number
              double t_start,
              double t_end);
    
    
    HD virtual Equity_prices * Get_new_prices(
               Equity_prices * eq_prices_in,
               double w,
               double delta_t
            );
};



#endif
