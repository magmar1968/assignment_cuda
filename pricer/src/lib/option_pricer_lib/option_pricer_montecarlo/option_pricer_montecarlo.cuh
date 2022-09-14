#ifndef __OPTION_PRICER_MONTECARLO__
#define __OPTION_PRICER_MONTECARLO__

#include "../option_pricer.cuh"
#include "../../path_gen_lib/path/path.cuh" //path
#include "../../contract_option_lib/contract_eq_option.cuh" //contract_eq_option
#include "../../support_lib/statistic_lib/statistic_lib.cuh" //average dev_std

#define HD __host__ __device__
class Option_pricer_montecarlo : public Option_pricer
{
  private:
    size_t _N; // number of montecarlo simulations
    double _price_square;
    double _price;
  public:
    //constructors & destructors 
    HD Option_pricer_montecarlo(){}
    HD Option_pricer_montecarlo(
                Contract_option *contract_option,
                Process         *process,
                size_t             N);
    HD ~Option_pricer_montecarlo(){}
    //getters & setters
    HD double Get_price() const;
    HD double Get_MonteCarlo_error() const;
    HD size_t   Get_N() const;

    HD void Set_N(size_t N);

    //functions
    HD void resimulate_option();
  private:
    HD void simulate_option();

    

    
};
#endif