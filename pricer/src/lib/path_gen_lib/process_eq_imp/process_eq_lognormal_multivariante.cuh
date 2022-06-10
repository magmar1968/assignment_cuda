#ifndef __PROCESS_EQ_LOGNORMAL_MULTIVARIANTE__
#define __PROCESS_EQ_LOGNORMAL_MULTIVARIANTE__

#include "process_eq_lognormal.cuh"

class Process_eq_lognormal_multivariante : public Process_eq_lognormal
{
  private:
    size_t _dim;
    double **correlation_matrix;
    Random_numbers * w_correlated;
  public:
    Process_eq_lognormal_multivariante(){}
    Process_eq_lognormal_multivariante(rnd::MyRandom * gnr,size_t dim);

    Random_numbers * Get_random_structure();
    Equity_prices  * Get_new_prices(Equity_prices * in,
                                    Random_numbers * w,
                                    double delta_t);

};


#endif