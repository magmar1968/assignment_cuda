#ifndef __CORRIDOR_PROBABILITY_CALCULATOR__
#define __CORRIDOR_PROBABILITY_CALCULATOR__

#include "../option_pricer.cuh"
#include "../../path_gen_lib/path/path.cuh" //path
#include "../../contract_option_lib/contract_eq_option.cuh" //contract_eq_option
#include "../../support_lib/statistic_lib/statistic_lib.cuh" //average dev_std

namespace prcr
{

  #define HD __host__ __device__
  class Corridor_probability_calculator : public Option_pricer
  {
    private:
      size_t _N; // number of montecarlo simulations
      double _P, _P2;
      

    public:
      //constructors & destructors 
      HD Corridor_probability_calculator(){}
      HD Corridor_probability_calculator(
                  Contract_eq_option_exotic_corridor* contract_option,
                  Process_eq_lognormal *process,
                  size_t             N);
      HD ~Corridor_probability_calculator(){}
      //getters & setters
      HD double Get_P() const;
      HD double Get_P2() const;
      HD size_t Get_N() const;

      HD void Set_N(size_t N);

      //functions
      HD void resimulate_option();
    private:
      HD void simulate_option();
      HD void compute_MC_error();
      
  };
}

#endif
