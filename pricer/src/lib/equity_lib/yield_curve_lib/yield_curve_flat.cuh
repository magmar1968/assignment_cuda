#ifndef __YIELD_CURVE_FLAT__
#define __YIELD_CURVE_FLAT__

#include "yield_curve.cuh"

class Yield_curve_flat : public Yield_curve
{
  private:
    double _rate;
  
  public:  
    //constructors and desctructors
    Yield_curve_flat(){}

    Yield_curve_flat(char * currency, double rate)
        :Yield_curve(currency),_rate(rate)
    {

    }

    ~Yield_curve_flat(){}
    //functions 
    double Get_spot_rate(double t)
    {
        return _rate;
    }

    double Get_forward_rate(double t_start, double t_end)
    {
        return _rate;
    }

};





#endif