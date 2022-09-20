#include "process_eq_lognormal.cuh"
namespace prcr
{
    
    __host__ __device__
    Process_eq_lognormal::Process_eq_lognormal(rnd::MyRandom * gnr,bool exact_solution,size_t l)
        :Process(gnr),_exact_solution(exact_solution),_l(l)
    {
    }

    __host__ __device__ udb
    Process_eq_lognormal::Get_new_equity_price(
                        Equity_description * eq_descr,
                        udb eq_price,
                        double w,
                        double t_start,
                        double t_end)
    {
        Yield_curve * yc = eq_descr -> Get_yc();
        double r = yc -> Get_forward_rate(t_start,t_end);

        double div_yield = eq_descr -> Get_dividend_yield();

        double sigma = eq_descr -> Get_vol_surface() -> Get_volatility(t_start,t_end);
        double delta_t = t_end - t_start;

        if ( _exact_solution)
            return compute_eq_price_exact(eq_price,r,div_yield,delta_t,w,sigma);
        else 
            return compute_eq_price_approximate(eq_price,r,div_yield,delta_t,w,sigma);

    }

    __host__ __device__ udb
    Process_eq_lognormal::compute_eq_price_exact(
                        udb eq_price,
                        double r,
                        double div_yield,
                        double delta_t,
                        double w,
                        double sigma)
    {
        return eq_price * exp( (r - div_yield - 0.5*sigma*sigma) *
            delta_t + sigma * sqrt(delta_t) * w ) ;
    }

    __host__ __device__ udb
    Process_eq_lognormal::compute_eq_price_approximate(
                        udb eq_price,
                        double r,
                        double div_yield,
                        double delta_t,
                        double w,
                        double sigma)
    {//sarebbe stra figo renderlo ricorsivo, ma non so come fare
        double new_delta_t = delta_t/static_cast<double>(_l);
        for( int i = 0; i < _l; ++i)
        {
            eq_price*= (1 + (r - div_yield)*new_delta_t + sigma* sqrt(delta_t)*w);
        }
        return eq_price;
    }

    __host__ __device__ Equity_prices * 
    Process_eq_lognormal::Get_new_prices(
                            Equity_prices  * eq_prices_in,
                            double           w,
                            double           delta_t)
    {
        Equity_prices* eq_prices_out = new Equity_prices;

        udb new_eq_price = Get_new_equity_price(
                            eq_prices_in -> Get_eq_description(),
                            eq_prices_in -> Get_eq_price(),
                            w ,
                            eq_prices_in -> Get_time(),
                            eq_prices_in -> Get_time() + delta_t); 

        eq_prices_out -> Set_eq_price(new_eq_price);
        eq_prices_out -> Set_time(eq_prices_in->Get_time() + delta_t);
        eq_prices_out -> Set_eq_description(eq_prices_in -> Get_eq_description());

        return eq_prices_out;
    }

}






