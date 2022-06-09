#include "stoch_exact.cuh"



namespace pricer
{
    HD StocProcess_ExactSolution::StocProcess_ExactSolution(double mu_0, double sigma_0, double S_0, double dt)
        : StochProcessImp(mu_0,sigma_0,S_0,dt)
    {
        _exact = true;
    }
    
    HD double StocProcess_ExactSolution::get_step(const double w)
    {
       return _S = _S * exp((_mu - (_sigma*_sigma) / 2.)*_dt + _sigma*sqrt(_dt)*w);
    }

}