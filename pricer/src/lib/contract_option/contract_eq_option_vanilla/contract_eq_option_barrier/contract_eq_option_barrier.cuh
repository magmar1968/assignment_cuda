#ifndef __CONTRACT_EQ_OPTION_BARRIER__
#define __CONTRACT_EQ_OPTION_BARRIER__

#include "../contract_eq_option_vanilla.cuh"


#define HD __host__ __device__


enum Barrier_type
{
    UP_IN,UP_OUT,DOWN_IN,DOWN_OUT
};

class Contract_eq_option_barrier : Contract_eq_option_vanilla
{
  private:
    double _barrier_level;
    Barrier_type _barrier_type;
  public:
    HD Contract_eq_option_barrier(){}
    HD Contract_eq_option_barrier(Equity_prices * eq_prices,
                                  Schedule      * schedule,
                                  double strike_price,
                                  char   contract_type,
                                  double barrier_level,
                                  Barrier_type b_type)
        :Contract_eq_option_vanilla(eq_prices,schedule,strike_price,contract_type),
        _barrier_level(barrier_level),_barrier_type(b_type)
        {}
    
    HD ~Contract_eq_option_barrier(){}

    //getters and setters 
    HD double Get_barrier_level()const{return _barrier_level;}
    HD Barrier_type Get_barrier_type()const{return _barrier_type;}

    HD void Set_barriel_level(const double b_level)
    {_barrier_level = b_level;}
    HD void Set_barrier_level(const Barrier_type b_type)
    {
        _barrier_type = b_type;
    }

    //functions

    HD double Pay_off(const Path * path);

    HD double Pay_off_up_in(const Path * path);
    HD double Pay_off_up_out(const Path * path);
    HD double Pay_off_down_in(const Path * path);
    HD double Pay_off_down_out(const Path * path);
};

#endif