#include "process_eq_lognormal.cuh"

__host__ __device__
Process_eq_lognormal::Process_eq_lognormal(rnd::MyRandom * gnr)
    :Process_eq(gnr)
{
}

__host__ __device__ pricer::udb
Process_eq_lognormal::Get_new_equity_price(
                    Equity_description * eq_descr,
                    pricer::udb eq_price,
                    double w,
                    double t_start,
                    double t_end)
{
    Yield_curve * yc = eq_descr -> Get_yc();
    double r = yc -> Get_forward_rate(t_start,t_end);

    double div_yield = eq_descr -> Get_dividend_yield();

    double sigma = eq_descr -> Get_vol_surface() -> Get_volatility(t_start,t_end);
    double delta_t = t_end - t_start;

    return eq_price * exp( (r - div_yield - 0.5*sigma*sigma) *
           delta_t + sigma * sqrt(delta_t) * w ) ;
}

__host__ __device__ Random_numbers *
Process_eq_lognormal::Get_random_structure()
{
        Random_numbers* w = NULL; 
        w->Set_element(0, Get_random_gaussian() ) ;
    return w ;
}


__host__ __device__ Equity_prices * 
Process_eq_lognormal::Get_new_prices(
                        Equity_prices  * eq_prices_in,
                        Random_numbers * w,
                        double delta_t)
{
    Equity_prices* eq_prices_out = NULL;

    pricer::udb new_eq_price = Get_new_equity_price(
                        eq_prices_in -> Get_eq_description(),
                        eq_prices_in -> Get_eq_price(),
                        w-> Get_element() ,
                        eq_prices_in -> Get_time(),
                        eq_prices_in -> Get_time() + delta_t); 

    eq_prices_out -> Set_eq_price(new_eq_price);
    eq_prices_out -> Set_time(eq_prices_in->Get_time() + delta_t);
    eq_prices_out -> Set_eq_description(eq_prices_in -> Get_eq_description());

    return eq_prices_out;
}







