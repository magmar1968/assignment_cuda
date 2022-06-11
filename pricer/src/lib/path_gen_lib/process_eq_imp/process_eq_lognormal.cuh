#ifndef __PROCESS_EQ_LOGNORMAL__
#define __PROCESS_EQ_LOGNORMAL__

#include "../process_eq.cuh"

class Process_eq_lognormal : public Process_eq
{
  public:
    Process_eq_lognormal(){};
    Process_eq_lognormal(rnd::MyRandom * gnr);
    virtual ~Process_eq_lognormal(){};
    //functions
    double Get_new_equity_price(
              Equity_description * eq_descr,
              double eq_price,
              double w,
              double t_start,
              double t_end);
    
    Random_numbers * Get_random_structure();
    virtual Equity_prices  * Get_new_prices(
              Equity_prices  * eq_prices_in,
              Random_numbers * w,
              double delta_t
            );
};



#endif