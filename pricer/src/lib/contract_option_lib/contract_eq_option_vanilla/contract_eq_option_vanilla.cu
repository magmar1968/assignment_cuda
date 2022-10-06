#include "contract_eq_option_vanilla.cuh"

namespace prcr
{

    __host__ __device__ double
    Contract_eq_option_vanilla::Pay_off(const Path *path)
    {
        double S_f = path -> Get_last_eq_price();
        return Pay_off_vanilla(S_f);
    }

    __host__ __device__ double
    Contract_eq_option_vanilla::Pay_off_vanilla(const double S_f)
    {
        

        switch (_contract_type)
        {
        case 'C':
            return Pay_off_vanilla_call(S_f);
        case 'P':
            return Pay_off_vanilla_put(S_f);
        default:
            return -1.;
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
}

