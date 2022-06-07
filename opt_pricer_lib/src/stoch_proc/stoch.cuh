#ifndef __STOCH_PROC__
#define __STOCH_PROC__
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
        HD virtual double get_dt() const = 0;
        HD virtual double get_exact() const = 0;
    };

    class StochProcessImp : public StochProcess
    {
    public:
        HD StochProcessImp(double mu_0, double sigma_0, double S_0, double dt);
        HD double getS() const;
        HD double get_dt() const;
        HD virtual double get_exact() const;

    protected:
        double _mu;
        double _sigma;
        double _S;
        double _dt;
        bool _exact;
    };

    
}

#endif