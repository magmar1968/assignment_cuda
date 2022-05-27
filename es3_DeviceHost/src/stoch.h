#ifndef __STOCH__
#define __STOCH__

#include <cmath>

class StochProcess
{
    public:
    StochProcess(double mu, double sigma);
    double EulerDoStep(double p, double dt, double random);
    double ExactDoStep(double dt);

    private:
    double _mu;
    double _sigma;
};


#endif