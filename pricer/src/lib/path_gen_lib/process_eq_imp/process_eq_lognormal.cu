#include "process_eq_lognormal.cuh"
namespace prcr
{
    
    __host__ __device__
    Process_eq_lognormal::Process_eq_lognormal(rnd::MyRandom * gnr,bool exact_solution,size_t l)
        :Process(gnr),_exact_solution(exact_solution),_l(l)
    {
    }

    __host__ __device__ double
    Process_eq_lognormal::Get_new_eq_price(
                        Equity_description * eq_descr,
                        double eq_price,
                        double w,
                        double delta_t)
    {
        double r = eq_descr -> Get_yc();
        

        double div_yield = eq_descr -> Get_dividend_yield();

        double sigma = eq_descr -> Get_vol_surface();
       // double delta_t = t_end - t_start;

        if ( _exact_solution)
            return compute_eq_price_exact(eq_price,r,div_yield,delta_t,w,sigma);
        else 
            return compute_eq_price_approximate(eq_price,r,div_yield,delta_t,w,sigma);

    }

    __host__ __device__ double
    Process_eq_lognormal::compute_eq_price_exact(
                        double eq_price,
                        double r,
                        double div_yield,
                        double delta_t,
                        double w,
                        double sigma)
    {
        return eq_price * exp( (r - div_yield - 0.5*sigma*sigma) *
            delta_t + sigma * sqrt(delta_t) * w ) ;
    }

    __host__ __device__ double
    Process_eq_lognormal::compute_eq_price_approximate(
                        double eq_price,
                        double r,
                        double div_yield,
                        double delta_t,
                        double w,
                        double sigma)
    {//sarebbe stra figo renderlo ricorsivo, ma non so come fare
        double new_delta_t = delta_t/static_cast<double>(_l);
        for( int i = 0; i < _l; ++i)
        {
            eq_price*= (1 + (r - div_yield)*new_delta_t + sigma* sqrt(new_delta_t)*w);
        }
        return eq_price;
    }

    /*__host__ __device__ Equity_prices*
    Process_eq_lognormal::Get_new_prices(
                            double  * eq_prices_in,
                            double           w,
                            double           delta_t)
    {
        double* eq_prices_out = new double;

        double new_eq_price = Get_new_equity_price(
                            eq_prices_in -> Get_eq_description(),
                            eq_prices_in -> Get_price(),
                            w ,
                            eq_prices_in -> Get_time(),
                            eq_prices_in -> Get_time() + delta_t); 

        return eq_prices_out;
    }*/

}






