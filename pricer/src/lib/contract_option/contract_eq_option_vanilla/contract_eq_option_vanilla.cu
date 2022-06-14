#include "contract_eq_option_vanilla.cuh"


__host__ __device__ double
Contract_eq_option_vanilla::Pay_off(const Path *path)
{
    return Pay_off_vanilla(path);
}

__host__ __device__ double
Contract_eq_option_vanilla::Pay_off_vanilla(const Path* path)
{
    size_t laste_step = path -> Get_dim();
    Equity_prices * final_price = path -> Get_equity_prices(laste_step);

    size_t n_eq = final_price -> Get_dim();
    double S_f = 0;

    for(size_t i = 0; i < n_eq; ++i)
    {
        S_f += final_price -> Get_eq_price(i);
    }

    switch (_contract_type)
    {
    case 'C':
        return Pay_off_vanilla_call(S_f);
        break;
    case 'P':
        return Pay_off_vanilla_put(S_f);
        break;
    default:
        return -1.;
        break;
    }
}

__host__ __device__ double
Contract_eq_option_vanilla::Pay_off_vanilla_call(const double S_f)   
{
    return  max(S_f - _strike_price,0.);
}

__host__ __device__ double 
Contract_eq_option_vanilla::Pay_off_vanilla_put(const double S_f)
{
    return max( _strike_price - S_f,0.);
}

