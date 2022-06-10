#ifndef __EQUITY_PRICES__
#define __EQUITY_PRICES__

#include "equity_description.cuh"

class Equity_prices
{
  private:
    double   _time;
    double * _prices;
    size_t   _dim;
    Equity_description ** _eq_descr;
  
  public:
    Equity_prices(){}
    Equity_prices(double time,
                  double * prices,
                  size_t   dim,
                  Equity_description ** equity_descriptions)
        :_time(time),_prices(prices),_dim(dim),
        _eq_descr(equity_descriptions)
    {}

    // getter & setter
    double Get_time()const{ return _time;}
    double Get_eq_price(const size_t i) const
    {
        if(i < _dim)
            return _prices[i];
        else
            exit(1); //forse non funzica su cuda
    }
    double Get_eq_price() const {return Get_eq_price(0);}
    double Get_dim() const {return _dim;}

    Equity_description * Get_eq_description(const size_t i) const
    {
        if(i < _dim)
            return _eq_descr[i];
        else
            exit(1); //forse non funzica su cuda
    }
    Equity_description * Get_eq_description() const {return Get_eq_description(0);}
    

    void Set_time(double t) {_time = t;}

    void Set_eq_price(size_t i, double price)
    {
        if ( i < _dim)
            _prices[i] = price;
        else
            exit(1); //forse non funzica su cuda
    }
    void Set_eq_price(double price){ Set_eq_price(0,price);}

    void Set_eq_description(size_t i , Equity_description * eq_description)
    {
        _eq_descr[i] = eq_description; 
    }
    void Set_eq_description(Equity_description * eq_description)
    {
        Set_eq_description(0,eq_description);
    }
};


#endif