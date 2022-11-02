#include "process_eq_binomial.cuh"
namespace prcr
{
    
    __host__ __device__
    Process_eq_binomial::Process_eq_binomial(rnd::MyRandom * gnr)
        :Process(gnr)
    {
    }

    __host__ __device__ double
    Process_eq_binomial::Get_new_eq_price(
                        Equity_description * eq_descr,
                        double eq_price,
                        double w,
                        double delta_t)
    {
        double r = eq_descr -> Get_yc();
        

        double div_yield = eq_descr -> Get_dividend_yield();

        double sigma = eq_descr -> Get_vol_surface();

       
        return compute_eq_price_exact(eq_price,r,div_yield,delta_t,w,sigma);
       
    }

    __host__ __device__ double
    Process_eq_binomial::compute_eq_price_exact(
                        double eq_price,
                        double r,
                        double div_yield,
                        double delta_t,
                        double w,
                        double sigma)
    {
        double z = 0;
        if(w >= 0)  z = 1;
        else z = -1;
        return eq_price * exp( (r - div_yield - 0.5*sigma*sigma) *
            delta_t + sigma * sqrt(delta_t) * z ) ;
    }

}






