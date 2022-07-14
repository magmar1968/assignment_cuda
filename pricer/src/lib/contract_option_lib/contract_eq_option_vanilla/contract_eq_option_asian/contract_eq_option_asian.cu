#include "contract_eq_option_asian.cuh"

__host__ __device__ double
Contract_eq_option_asian::Pay_off(const Path * path)
{
    double mean_S = 0.;

    uint n_step = path->Get_dim() - path->Get_start_ind(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        mean_S += path->Get_equity_prices(i).Get_eq_price().get_number();
        
    }
    mean_S = mean_S / (double)n_step;
    return Pay_off_vanilla(mean_S);
}