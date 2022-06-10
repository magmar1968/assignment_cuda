#ifndef __VOLATILITY__
#define __VOLATILITY__

class Volatility_surface
{
  private:
    double _vol;  
  
  public:
    //constructor & destructor
    Volatility_surface() {}
    Volatility_surface( double vol)
        :_vol(vol)
    {}
    // getter
    double Get_volatility(double t_start, double t_end) const
    {
        return _vol;
    }

};

#endif