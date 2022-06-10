#ifndef __EQUITY_PRICES__
#define __EQUITY_PRICES__

#include "equity_description.cuh"

class Equity_prices
{
  private:
    double   _time;
    double * _prices;
    size_t   _dim;
    Equity_description ** _equity_descr;
  
  public:
    Equity_prices(){}
    Equity_prices(double time,
                  double * prices,
                  size_t   dim,
                  Equity_description ** equity_descriptions)
        :_time(time),_prices(prices),_dim(dim),
        _equity_descr(equity_descriptions)
    {}

    // getter & setter
    double Get_time()const{ return _time;}
    double Get_price(const size_t i) const
    {
        if(i < _dim)
            return _prices[i];
        else
            exit(1); //forse non funzica su cuda
    }
    double Get_dim() const {return _dim;}
    Equity_description * Get_eq_description(const size_t i) const
    {
        if(i < _dim)
            return _equity_descr[i];
        else
            exit(1); //forse non funzica su cuda
    }
    

    void Set_time(double t) {_time = t;}

    void Set_price(size_t i, double price)
    {
        if ( i < _dim)
            _prices[i] = price;
        else
            exit(1); //forse non funzica su cuda
    }
};


#endif