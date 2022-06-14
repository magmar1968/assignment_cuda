#ifndef __YIELD_CURVE_FLAT__
#define __YIELD_CURVE_FLAT__

#include "yield_curve.cuh"

//cuda macro
#define H  __host__
#define D  __device__
#define HD __host__ __device__


class Yield_curve_flat : public Yield_curve
{
  private:
    double _rate;
  
  public:  
    //constructors and desctructors
    HD Yield_curve_flat(){}

    HD Yield_curve_flat(char * currency, double rate)
        :Yield_curve(currency),_rate(rate)
    {

    }

    HD ~Yield_curve_flat(){}
    //functions 
    HD double Get_spot_rate(double t) const
    {
        return _rate;
    }

    HD double Get_forward_rate(double t_start, double t_end) const
    {
        return _rate;
    }

};





#endif