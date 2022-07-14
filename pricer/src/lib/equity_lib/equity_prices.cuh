/**
 * @file equity_prices.cuh
 * @author Lorenzo Magnoni - Andrea Ripamonti
 * @brief  Tiene traccia del prezzo di un determinato numero (_dim) di sottostanti
 *         al tempo _time. I sottostanti interessati sono contenuti nel vettore 
 *         _eq_descr. La classe può lavorare sia su device che host
 * 
 */
#ifndef __EQUITY_PRICES__
#define __EQUITY_PRICES__

#include "equity_description.cuh"
#include "../support_lib/myDouble_lib/myudouble.cuh"

#define HD __host__ __device__


class Equity_prices
{
  private:
    double   _time;      //current time
    pricer::udb _price;    //current price
    Equity_description *_eq_descr;
  
  public:
    //constructors & destructors
    HD Equity_prices(){} ;
    HD Equity_prices(double time,
                  pricer::udb price,
                  Equity_description* equity_description)
        :_time(time)
    {
        _price = price;
        _eq_descr = equity_description;
    }
    HD ~Equity_prices()
    {}

    // getter & setter
    HD double Get_time()const{ return _time;}
    HD pricer::udb Get_eq_price() const
    {
        return _price;
    }

    HD Equity_description* Get_eq_description() const
    {
        return _eq_descr;
    }
    
    //setters
    HD void Set_time(double t) {_time = t;}

    HD void Set_eq_price(pricer::udb price)
    {
        _price = price;
    }

    HD void Set_eq_description( Equity_description * eq_description)
    {
        _eq_descr= eq_description; 
    }
};


#endif
