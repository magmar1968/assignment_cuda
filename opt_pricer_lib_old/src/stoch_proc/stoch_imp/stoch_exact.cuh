#ifndef __STOCH_EXACTSOLUTION__
#define __STOCH_EXACTSOLUTION__

#include "../stoch.cuh"

namespace pricer
{
    class StocProcess_ExactSolution : public StochProcessImp
    {
    public:
        HD StocProcess_ExactSolution(double mu_0, double sigma_0, double S_0, double dt);
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };
}



#endif