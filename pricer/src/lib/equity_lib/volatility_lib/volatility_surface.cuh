/**
 * @file volatility_surface.cuh
 * @author Lorenzo Magnoni - Andrea Ripamonti
 * @brief Descrive una generica superficie di volatilità. Al momento è disponibile
 *        solo il caso banale di una curva piatta. La classe gira sia su host che 
 *        device.
 */
#ifndef __VOLATILITY__
#define __VOLATILITY__

#define H  __host__
#define D  __device__
#define HD __host__ __device__


class Volatility_surface
{
  private:
    double _vol;  
  
  public:
    //constructor & destructor
    HD Volatility_surface() {}
    HD Volatility_surface( double vol)
        :_vol(vol)
    {}
    // getter
    HD double Get_volatility(double t_start, double t_end) const
    {
        return _vol;
    }
    HD double Get_volatility()
    {
        return _vol;
    }
       
};

#endif