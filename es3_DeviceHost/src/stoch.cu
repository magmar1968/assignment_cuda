#include "stoch.h"

StochProcess::StochProcess(double mu, double sigma)
{
    _mu = mu;
    _sigma = sigma;
}

double StochProcess::EulerDoStep(double p, double dt, double random)
{
    return p * (_mu * dt + _sigma * random * sqrt(dt));
    
}