/**
 * @file equity_prices.cuh
 * @author Lorenzo Magnoni - Andrea Ripamonti
 * @brief  Tiene traccia del prezzo di un sottostante al tempo _time.
 *         Il sottostante interessato è contenuto in eq_descr. 
 *         La classe può lavorare sia su device che host
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
    double               _time;      //current time
    pricer::udb          _price;    //current price
    Equity_description * _eq_descr;   
  
  public:
    //constructors & destructors
    HD Equity_prices(){} ;
    HD Equity_prices(double time,
                    pricer::udb  price,                               
                    Equity_description * equity_descriptions)
        :_time(time)
    {
        _price = price;
        _eq_descr = equity_descriptions; 
    }
    HD Equity_prices()
    {
        _eq_descr = new Equity_description;
    }
    HD ~Equity_prices()
    {
        delete(_eq_descr); 
    }

    // getter & setter
    HD double      Get_time()     const{ return _time;}
    HD pricer::udb Get_eq_price() const{ return _price;}
    HD Equity_description * Get_eq_description() const // ha senso?
    {
        return _eq_descr;
    }    

    //setters
    HD void Set_time(double t)             { _time  = t;}
    HD void Set_eq_price(pricer::udb price){ _price = price;}
    HD void Set_eq_description( Equity_description * eq_description)
    {
        _eq_descr = eq_description; 
    }
};


#endif
