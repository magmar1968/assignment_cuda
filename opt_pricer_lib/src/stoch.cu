#include "../include/stoch.h"  //cambiare in hpp

namespace pricer
{
    HD StochProcessImp::StochProcessImp(double mu_0, double sigma_0, double S_0, double dt)
        : _mu(mu_0), _sigma(sigma_0), _S(S_0), _dt(dt)
    {
    }



    HD double ExactSolution::get_step(const double w)
    {
       return _S = _S * exp((_mu - (_sigma*_sigma) / 2.)*dt + _sigma*sqrt(_dt)*w);
    }


    HD double EulerSolution::get_step(const double w)
    {
        return _S = _S * (_mu * dt + _sigma * sqrt(dt) * w *);
    }



}