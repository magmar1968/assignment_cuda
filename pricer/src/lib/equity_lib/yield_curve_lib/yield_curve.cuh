/**
 * @file yield_curve.cuh
 * @author Lorenzo Magnoni - Andrea Ripamonti
 * @brief  Classe template per la gestione e descrizione dei tassi di interesse.
 *         La classe gira sia su host che device
 * 
 */

#ifndef __YIELD_CURVE__
#define __YIELD_CURVE__


namespace
{

#define H  __host__
#define D  __device__
#define HD __host__ __device__

class Yield_curve
{
  private:
    char * _currency;
 
  public:
    //constructors & destructors
    Yield_curve(){};
    Yield_curve( char * currency)
        :_currency(currency)
    {
    }
    virtual ~Yield_curve(){}

    virtual double Get_spot_rate(double t) const = 0;
    virtual double Get_forward_rate(double t_start,
                                    double t_end) const = 0;
    
    double Get_discount_factor(double t) const 
    {
        return exp(- Get_spot_rate(t)*t);
    }
    
    double Get_discount_factor(double t_start,
                               double t_end ) const
    {
        return exp(- Get_forward_rate(t_start,t_end) * (t_start - t_end));
    }

};
}

#endif