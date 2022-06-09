#ifndef __OPTION_PRICES__
#define __OPTION_PRICES__

/*Option prices implementation*/
#include "option_prices.cuh"
namespace pricer
{

    Option_prices::Option_prices(double time_init, double* prices_init, int dim_init)
    {
        Set_time(time_init);
        dim = dim_init;
        prices = new double[dim];
        for(int i=0;i<dim;i++)
        {
            Set_price(i,prices_init[i]);
        }
    }
    double Option_prices::Get_time(void)
            {
                return time;
            }
    void Option_prices::Set_time(double time_init)
            {
                time = time_init;
            }
    double Option_prices::Get_price(int i)
            {
                if(i>=0 && i<dim)
                {
                    return prices[i] ;
                }
                else
                {
                    return -1 ;
                }
            }
    int Option_prices::Set_price(int i, double price_init)
            {
                if((i>=0)&&(i<dim))
                {
                    prices[i] = price_init;
                    return 0;// boh ma un setter può avere un return??
                }
                else
                {
                    return -1;
                }
            }
}

#endif