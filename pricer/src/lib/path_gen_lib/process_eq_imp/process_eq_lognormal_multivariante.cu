#include "process_eq_lognormal_multivariante.cuh"



__host__ __device__
Process_eq_lognormal_multivariante::Process_eq_lognormal_multivariante
                                    (rnd::MyRandom * gnr, size_t dim)
    :Process_eq_lognormal(gnr),_dim(dim)
{}

__host__ __device__ Random_numbers * 
Process_eq_lognormal_multivariante::Get_random_structure()
{
    Random_numbers * w = new Random_numbers(_dim);
    for( size_t i = 0; i<_dim; ++i)
    {
        w -> Set_element(i, Get_random_gaussian());
    }
    return w;
}

__host__ __device__ Equity_prices * 
Process_eq_lognormal_multivariante::Get_new_prices(
                            Equity_prices  * eq_prices_in,
                            Random_numbers * w,
                            double delta_t)
{
    Equity_prices * eq_prices_out = new Equity_prices();

    //Generate_corellations(w_correlated, w); da implementare
    w_correlated = w;

    for (size_t i = 0; i < _dim; ++i)
    {
        double new_eq_price = Get_new_equity_price(
                        eq_prices_in -> Get_eq_description(i),
                        eq_prices_in -> Get_eq_price(i),
                        w_correlated -> Get_element(i),
                        eq_prices_in -> Get_time(),
                        eq_prices_in -> Get_time() + delta_t);
        eq_prices_out->Set_eq_price(i, new_eq_price);
        eq_prices_out->Set_time(eq_prices_in -> Get_time() + delta_t);
        eq_prices_out->Set_eq_description(i, eq_prices_in -> Get_eq_description(i));
    }
    return eq_prices_out;
}
