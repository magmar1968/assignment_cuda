#ifndef __STOCH_EULERSOLUTION__
#define __STOCH_EULERSOLUTION__

#include "../stoch.cuh"

namespace pricer
{
    class StocProcess_EulerSolution : public StochProcessImp
    {
    public:
        HD StocProcess_EulerSolution(double mu_0, double sigma_0, double S_0, double dt);
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };
}

#endif