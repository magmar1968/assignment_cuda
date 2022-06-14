/**
 * @file equity_description.cuh
 * @author Lorenzo Magnoni - Andrea Ripamonti
 * @brief Questa classe è dedicata alla descrizione di un generico sottostante
 *        azionario. Ne contiente quindi tutte le informazioni per avere una 
 *        completa descrizione di quest'ultimo.
 * 
 */


#ifndef __EQUITY_DESCRIPTION__
#define __EQUITY_DESCRIPTION__

#include "yield_curve_lib/yield_curve.cuh"
#include "volatility_lib/volatility_surface.cuh"

#define H __host__
#define D __device__
#define HD __host__ __device__

class Equity_description
{
  public:
    //constructors
    HD Equity_description(){}

    HD Equity_description( char *  isin_code,
                        char *  name,
                        char *  currency,
                        double  dividend_yield,
                        Yield_curve * yield_curve,
                        Volatility_surface * volatility_surface)
      :_isin_code(isin_code), _name(name), _currency(currency),
       _dividend_yield(dividend_yield),_yc(yield_curve),
       _vol_surf(volatility_surface)
    {

    }
    HD virtual ~Equity_description(){}

    //getters
    HD char * Get_isin_code()const {return _isin_code; }
    HD char * Get_name()     const {return _name;}
    HD char * Get_currency() const {return _currency;}
    HD double Get_dividend_yield() const {return _dividend_yield;}
    HD Yield_curve * Get_yc()const {return _yc;}   
    HD Volatility_surface * Get_vol_surface()const{return _vol_surf;}   

    //setters
    HD void Set_isin_code(char * isin_code){_isin_code = isin_code;}
    HD void Set_name(char * name)          {_name = name;}
    HD void Set_currency(char * currency)  {_currency = currency;}
    HD void Set_dividend_yield(double yield){_dividend_yield = yield;}
    HD void Set_yc(Yield_curve * yc){_yc = yc;}
    HD void Set_vol_surface(Volatility_surface * vol_surf)
    {_vol_surf = vol_surf;}

  private:
    char * _isin_code;
    char * _name;
    char * _currency;
    double _dividend_yield;
    Yield_curve * _yc;
    Volatility_surface * _vol_surf;

};

#endif