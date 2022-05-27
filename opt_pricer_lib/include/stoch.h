#ifndef __STOCH__
#define __STOCH__

#include <cmath>

namespace pricer
{
#define __host__ H;
#define __device__ D;
#define __host__ __device__ HD;

    class StochProcess
    {
    public:
        HD StochProcess();
        HD virtual double get_step() = 0;  //implementare per sottostanti multiple
    };

    class StochProcessImp : public StochProcess
    {
    public:
        HD StochProcessImp(double mu_0, double sigma_0, double S_0, double dt);

    protected:
        double _mu;
        double _sigma;
        double _S;
        double _dt;
    };



    class ExactSolution : public StochProcess
    {
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };


    class EulerSolution : public StochProcess
    {
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };
}

#endif