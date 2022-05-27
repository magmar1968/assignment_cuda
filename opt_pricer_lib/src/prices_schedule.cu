#include "prices_schedule.hpp"
namespace pricer
{
    /*#define H __host__
    #define D __device__
    #define HD __host__ __device__ */

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
                }
                else
                {
                    return -1;
                }
            }


 /*--------------------------------------------------------------------------------------------------------------------*/



    Schedule::Schedule(double t_ref, double delta_t, int dim_init)
    {
        dim = dim_init;
        t = new double[dim];
        for(int i=0;i<dim;i++)
        {
            t[i] = t_ref +delta_t * i;
        }
        assert(Check_order());
    }
    Schedule::Schedule(double *t_init, int dim_init)
    {
        dim = dim_init;
        t = new double[dim];
        for(int i=0;i<dim;i++)
        {
            t[i] = t_init[i];
        }
        assert(Check_order());
    }
    double Schedule::Get_t(int i)
    {
        return t[i];
    }
    int Schedule::Get_dim(void)
    {
        return dim;
    }
    bool Schedule::Check_order()
        {
            for(int i=1;i<dim;i++)
            {
                if(t[i]<=t[i-1]) {return false;}
            }
            return true;
        }
}


/*int main()   //per fare qualche prova
{
    int dim = 5;
    double a[5] = {12,54,64,67,68};
    double time = 6;
    pricer::Option_prices c(5, &a[0], 5);
    std::cout<<c.Get_price(3)<<std::endl;

    return 0;
}*/
