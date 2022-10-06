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

namespace prcr
{

  #define H __host__


  class Equity_description
    
  {
    private:
      double _dividend_yield;
      double _yc;
      double _vol_surf;

    public:
      //constructors
      HD Equity_description(){}

      HD Equity_description( 
                          double  dividend_yield,
                          double  yield_curve,
                          double  volatility_surface)
        :_dividend_yield(dividend_yield),_yc(yield_curve),
        _vol_surf(volatility_surface)
      {    }
      HD virtual ~Equity_description(){}

      //getters
      HD double Get_dividend_yield() const {return _dividend_yield;}
      HD double Get_yc()const {return _yc;}   
      HD double Get_vol_surface()const{return _vol_surf;}   

      //setters
      HD void Set_dividend_yield(double yield){_dividend_yield = yield;}
      HD void Set_yc(double yc)               {_yc = yc;               }
      HD void Set_vol_surface(double vol_surf){_vol_surf = vol_surf;   }


  };
}

#endif