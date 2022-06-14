#ifndef __PROCESS_EQ_LOGNORMAL__
#define __PROCESS_EQ_LOGNORMAL__

#include "../process_eq.cuh"


// cuda macro
#define H __host__
#define D __device__
#define HD __host__ __device__ 


class Process_eq_lognormal : public Process_eq
{
  public:
    HD Process_eq_lognormal(){};
    HD Process_eq_lognormal(rnd::MyRandom * gnr);
    HD virtual ~Process_eq_lognormal(){};
    //functions
    HD double Get_new_equity_price(
              Equity_description * eq_descr,
              double eq_price,
              double w,
              double t_start,
              double t_end);
    
    HD Random_numbers * Get_random_structure();
    HD Equity_prices* Get_new_prices(
              Equity_prices  * eq_prices_in,
              Random_numbers * w,
              double delta_t
            );
};



#endif