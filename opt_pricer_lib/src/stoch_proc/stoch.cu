#include "stoch.cuh"  

namespace pricer
{
    HD StochProcessImp::StochProcessImp(double mu_0, double sigma_0, double S_0, double dt)
        : _mu(mu_0), _sigma(sigma_0), _S(S_0), _dt(dt)
    {
    }

    HD double StochProcessImp::getS() const
    {
         return _S;
    }    

    HD double StochProcessImp::get_dt() const
    {
        return _dt;
    }
}