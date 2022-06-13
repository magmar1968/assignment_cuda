#ifndef __YIELD_CURVE_TERM_STRUCTURE__
#define __YIELD_CURVE_TERM_STRUCTURE__


#include "yield_curve.cuh"

#define H  __host__
#define D  __device__
#define HD __host__ __device__

class Yield_curve_term_structure : public Yield_curve
{
  private:
    double * _rates, * _times;
    int      _dim;
  
  public:
    //constructor & destructors
    HD Yield_curve_term_structure(){}

    HD Yield_curve_term_structure(char   * currency_init,
                               double * rate_deposit,
                               double * t_deposit,
                               int      num_deposit,
                               double * rate_swap,
                               double * t_swap,
                               int      num_swap )
    {
        // to implement
    }

    HD Yield_curve_term_structure(char   * currency,
                               double * rates, 
                               double * times,
                               int      dim)
        :Yield_curve(currency),_rates(rates),
         _times(times),_dim(dim)
    { }

    HD ~Yield_curve_term_structure(){}

    //functions

    HD double Get_spot_rate(double t)
    {
        for(size_t i = 1; i < _dim; ++i)
        {
            if (_times[i] == t)
            {
                return _rates[i];
            }
            else if( _times[i] > t)
            {
                // possibile implementare funzioni di interpolazione più
                // complesse
                return (_rates[i] + _rates[i -1])/2. ;
            }
        }
    }

    HD double Get_forward_rate(double t_start, double t_end)
    {
        return Get_spot_rate(t_end - t_start); //corretto?? not sure
    } 

};




#endif