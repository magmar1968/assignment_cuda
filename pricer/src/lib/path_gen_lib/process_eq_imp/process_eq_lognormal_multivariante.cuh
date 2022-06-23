#ifndef __PROCESS_EQ_LOGNORMAL_MULTIVARIANTE__
#define __PROCESS_EQ_LOGNORMAL_MULTIVARIANTE__

#include "process_eq_lognormal.cuh"
#include "../../support_lib//myDouble_lib/myudouble.cuh"

#define H  __host__
#define D  __device__
#define HD __host__ __device__

class Process_eq_lognormal_multivariante : public Process_eq_lognormal
{
  private:
    size_t _dim;
    double **correlation_matrix;
    Random_numbers * w_correlated;
  public:
    HD Process_eq_lognormal_multivariante(){}
    HD Process_eq_lognormal_multivariante(rnd::MyRandom * gnr,size_t dim);

    HD Random_numbers * Get_random_structure();
    HD Equity_prices  * Get_new_prices(Equity_prices * in,
                                    Random_numbers * w,
                                    double delta_t);

};


#endif