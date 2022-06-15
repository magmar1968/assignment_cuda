#include "contract_eq_option_asian.cuh"

__host__ __device__ double
Contract_eq_option_asian::Pay_off(Path * path)
{
    double S = 0.;
    double mean_S = 0.;

    size_t n_eq = path->Get_starting_point()->Get_dim();
    uint n_step = path->Get_dim() - path->Get_start_ind(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        S = 0.;
        Equity_prices *  current_eq = path->Get_equity_prices(i); 
        for(size_t j = 0; j < n_eq; ++j)
        {
            S += current_eq ->Get_eq_price();
        }    
        mean_S += S/(double)n_step;
    }

    return Pay_off_vanilla(mean_S);
}