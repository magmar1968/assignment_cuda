#ifndef __PAY_OFF__
#define __PAY_OFF__

#include <math.h>  //max
#include "lib.hpp" //average
#include <vector>

namespace pricer
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    
    typedef unsigned int uint; //for windows user

    class PayOff
    {
    public:
        HD PayOff();
        HD ~PayOff();

        virtual HD double getPayOff_call() = 0;
        virtual HD double getPayOff_put() = 0;
    
    };

    class PayOff_vanilla : public PayOff
    {
    public:
        // HD PayOff_vanilla();
        HD PayOff_vanilla(double f_S, double E);
        HD ~PayOff_vanilla();

        virtual HD double getPayOff_call();
        virtual HD double getPayOff_put();
    protected:
        double   _f_S,_E;
    };

    class PayOff_digital : public PayOff_vanilla
    {
    public:
        HD PayOff_digital(double f_S, double E, double K);

        HD double getPayOff_call();
        HD double getPayOff_put();
    private:
        double _K;
    };

    class PayOff_esotic : public PayOff
    {
    public:

        HD PayOff_esotic(const std::vector<double> & path, double E);
        HD ~PayOff_esotic();

        virtual HD double getPayOff_call();
        virtual HD double getPayOff_put();

    protected:
        const std::vector<double> &_path;
        size_t   _steps;
        double   _E;
    };

    class PayOff_asiatic : public PayOff_esotic
    {
    public:
        HD double getPayOff_call();
        HD double getPayOff_put();
    private:
        bool   _avg_not_computed = true;
        double _avg;
    };

    
}




#endif