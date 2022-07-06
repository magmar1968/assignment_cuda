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
    pricer::udb * _prices;    //current price
    size_t   _dim;       //# equities
    Equity_description ** _eq_descr;
  
  public:
    //constructors & destructors
    HD Equity_prices(){} ;
    HD Equity_prices(double time,
                  pricer::udb * prices,
                  size_t   dim,
                  Equity_description ** equity_descriptions)
        :_time(time),_dim(dim)
    {
        _prices = prices;
        _eq_descr = equity_descriptions;
    }
    HD Equity_prices(size_t dim)
        :_dim(dim)
    {
        _prices = new pricer::udb[_dim];
        _eq_descr = new Equity_description*[_dim];
    }
    HD ~Equity_prices()
    {
	    delete[](_prices);
	    //delete[](_eq_descr);
    }

    // getter & setter
    HD double Get_time()const{ return _time;}
    HD pricer::udb Get_eq_price(const size_t i) const
    {
        if(i < _dim)
            return _prices[i];
        else
            return -100;// exit(1); //forse non funzica su cuda
    }
    HD pricer::udb Get_eq_price() const {return Get_eq_price(0);}
    HD size_t Get_dim() const {return _dim;}

    HD Equity_description * Get_eq_description(const size_t i) const
    {
        if (i < _dim)
            return _eq_descr[i];
        else
            return NULL; //exit(1); //forse non funzica su cuda
    }
    HD Equity_description * Get_eq_description() const {return Get_eq_description(0);}
    
    //setters
    HD void Set_time(double t) {_time = t;}

    HD void Set_eq_price(size_t i, pricer::udb price)
    {
        if (i < _dim)
            _prices[i] = price;
        else
           ;// exit(1); //forse non funzica su cuda
    }
    HD void Set_eq_price(pricer::udb price){ Set_eq_price(0,price);}

    HD void Set_eq_description(size_t i , Equity_description * eq_description)
    {
        _eq_descr[i] = eq_description; 
    }
    HD void Set_eq_description(Equity_description * eq_description)
    {
        Set_eq_description(0,eq_description);
    }
};


#endif
