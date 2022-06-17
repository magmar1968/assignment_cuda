#ifndef __CONTRACT_EQ_OPTION__
#define __CONTRACT_EQ_OPTION__

#include "contract_option.cuh"
#include "../equity_lib/equity_prices.cuh"
#include "../path_gen_lib/path/path.cuh"

#define HD __host__ __device__

class Contract_eq_option : public Contract_option
{
  private:
    Equity_prices * _eq_prices;

  public:
    //constructor & destructors
    HD Contract_eq_option(void) {}
    HD Contract_eq_option(
                       Equity_prices *equity_prices,
                       Schedule      *schedule)
        :Contract_option(schedule),_eq_prices(equity_prices)
        {}

    HD virtual ~Contract_eq_option(void){}
    
    HD virtual double Pay_off(const Path *path){return 0.;} ;

    HD Equity_prices * Get_eq_prices() const
    {return _eq_prices;}
} ;







#endif