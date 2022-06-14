#ifndef __CONTRACT_EQ_OPTION__
#define __CONTRACT_EQ_OPTION__

#include "contract_option.cuh"
#include "../equity_lib/equity_prices.cuh"
#include "../path_gen_lib/path/path.cuh"

class Contract_eq_option : public Contract_option
{
  private:
    Equity_prices * _equity_prices;

  public:
    //constructor & destructors
    Contract_eq_option(void) {}
    Contract_eq_option(Equity_prices *equity_prices,
                       Schedule      *schedule)
        :Contract_option(schedule),_equity_prices(equity_prices)
        {}

    virtual ~Contract_eq_option(void){}
    
    virtual double Pay_off(const Path *path) = 0 ;
} ;







#endif