#ifndef __PROCESS_EQ_LOGNORMAL__
#define __PROCESS_EQ_LOGNORMAL__

#include "../process.cuh"
#include "../../support_lib/myDouble_lib/myudouble.cuh"
#include "../../equity_lib/equity_prices.cuh"

// cuda macro

namespace prcr
{
#define H __host__
#define D __device__
#define HD __host__ __device__ 
  
  class Process_eq_lognormal : public Process
  {
    private:
      bool _exact_solution;
      size_t _l;
    public:
      HD Process_eq_lognormal(){};
      HD Process_eq_lognormal(rnd::MyRandom * gnr, 
                              bool exact_solution = true,
                              size_t l = 1); //n extra step for the euler scheme);
      HD virtual ~Process_eq_lognormal(){};
      //functions
      HD udb Get_new_equity_price(
                Equity_description * eq_descr,
                udb eq_price,
                double w, //random number
                double t_start,
                double t_end);
      
      
      HD virtual Equity_prices Get_new_prices(
                Equity_prices * eq_prices_in,
                double w,
                double delta_t
              );

    HD void Set_to_approximate_solution(){_exact_solution = false;}
    HD void Set_to_exact_solution()      {_exact_solution = true; }
    //functions 
    private:
      HD udb compute_eq_price_exact(
                      udb eq_price,
                      double r,
                      double div_yield,
                      double delta_t,
                      double w,
                      double sigma);
      HD udb compute_eq_price_approximate(
                      udb eq_price,
                      double r,
                      double div_yield,
                      double delta_t,
                      double w,
                      double sigma);


  };

} // namespace prcr



#endif
