#include "contract_eq_option_barrier.cuh"

__host__ __device__ double 
Contract_eq_option_barrier::Pay_off(const Path * path)
{
    switch (_barrier_type)
    {
    case Barrier_type::UP_IN:
        return Pay_off_up_in(path);
        break;
    case Barrier_type::UP_OUT:
        return Pay_off_up_out(path);
        break;
    case Barrier_type::DOWN_IN:
        return Pay_off_down_in(path);
        break;
    case Barrier_type::DOWN_OUT:
        return Pay_off_down_out(path);
    default:
        exit(3);
        break;
    }
}


__host__ __device__ double
Contract_eq_option_barrier::Pay_off_up_in(const Path * path)
{
    double S = 0;
    size_t n_eq = path->Get_starting_point()->Get_dim(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        S = 0.;
        Equity_prices *  current_eq = path->Get_equity_prices(i); 
        for(size_t j = 0; j < n_eq; ++i )
        {
            S += current_eq->Get_eq_price(j);
        }
        if(S >= _barrier_level)
            return Pay_off_vanilla(path);
    }
    return 0;// nel caso il sottostante non tocchi la barriera
}

__host__ __device__ double
Contract_eq_option_barrier::Pay_off_up_out(const Path * path)
{
    double S = 0;
    size_t n_eq = path->Get_starting_point()->Get_dim(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        S = 0.;
        Equity_prices *  current_eq = path->Get_equity_prices(i); 
        for(size_t j = 0; j < n_eq; ++i )
        {
            S += current_eq->Get_eq_price(j);
        }
        if(S >= _barrier_level)
            return 0;
    }
    return Pay_off_vanilla(path);// nel caso il sottostante non tocchi la barriera
}

__host__ __device__ double
Contract_eq_option_barrier::Pay_off_down_in(const Path * path)
{
    double S = 0;
    size_t n_eq = path->Get_starting_point()->Get_dim(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        S = 0.;
        Equity_prices *  current_eq = path->Get_equity_prices(i); 
        for(size_t j = 0; j < n_eq; ++i )
        {
            S += current_eq->Get_eq_price(j);
        }
        if(S <= _barrier_level)
            return Pay_off_vanilla(path);
    }
    return 0;// nel caso il sottostante non tocchi la barriera
}

__host__ __device__ double
Contract_eq_option_barrier::Pay_off_down_out(const Path * path)
{
   double S = 0;
    size_t n_eq = path->Get_starting_point()->Get_dim(); 
    for(size_t i = path->Get_start_ind(); i < path->Get_dim(); ++i)
    {
        S = 0.;
        Equity_prices *  current_eq = path->Get_equity_prices(i); 
        for(size_t j = 0; j < n_eq; ++i )
        {
            S += current_eq->Get_eq_price(j);
        }
        if(S <= _barrier_level)
            return 0;
    }
    return Pay_off_vanilla(path);// nel caso il sottostante non tocchi la barriera
}