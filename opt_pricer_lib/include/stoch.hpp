#ifndef __STOCH__
#define __STOCH__
#include <math.h>

namespace pricer
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__   

    class StochProcess
    {
    public:
        HD StochProcess(){};
        HD virtual double get_step(const double ) = 0;  //implementare per sottostanti multiple
        HD virtual double getS() const = 0;
    };

    class StochProcessImp : public StochProcess
    {
    public:
        HD StochProcessImp(double mu_0, double sigma_0, double S_0, double dt);
        HD double getS() const;

    protected:
        double _mu;
        double _sigma;
        double _S;
        double _dt;
    };



    class ExactSolution : public StochProcessImp
    {
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };


    class EulerSolution : public StochProcessImp
    {
    public:
        HD EulerSolution(double mu_0, double sigma_0, double S_0, double dt);
        HD double get_step(const double w);
        //HD double get_step(double mu, double sigma);            //se mu e sigma variano
    };
}

#endif