#ifndef __CONTRACT_EQ_OPTION_ASIAN__
#define __CONTRACT_EQ_OPTION_ASIAN__

#include "../contract_eq_option_vanilla.cuh"

#define HD __host__ __device__

class Contract_eq_option_asian : Contract_eq_option_vanilla
{
private:
    
public:
    HD Contract_eq_option_asian(){}
    HD Contract_eq_option_asian(Equity_prices * eq_prices,
                                  Schedule      * schedule,
                                  double strike_price,
                                  char   contract_type)
        :Contract_eq_option_vanilla(eq_prices,schedule,strike_price, contract_type)
        {}
    HD ~Contract_eq_option_asian(){};

    HD virtual double Pay_off(Path * path);
};



#endif