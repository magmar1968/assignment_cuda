#include "stoch_euler.cuh"

namespace pricer
{
    HD StocProcess_EulerSolution::StocProcess_EulerSolution(double mu_0, double sigma_0, double S_0, double dt)
        : StochProcessImp(mu_0,sigma_0,S_0,dt)
    {
        
    }

    HD double StocProcess_EulerSolution::get_step(const double w)
    {
        return _S = _S * exp(_mu * _dt + _sigma * sqrt(_dt) * w );
    }


}