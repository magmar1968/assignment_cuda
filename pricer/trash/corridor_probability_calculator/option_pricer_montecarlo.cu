#include "corridor_probability_calculator.cuh"

namespace prcr
{

    __host__ __device__
        Corridor_probability_calculator::Corridor_probability_calculator(
                    Contract_eq_option_exotic_corridor* contract_option,
                    Process_eq_lognormal *process,
                    size_t             N)
            :Option_pricer(contract_option,process),_N(N)
    {
        simulate_option();
    }

    __host__ __device__ double 
    Option_pricer_montecarlo::Get_P() const
    {
        return _P2;
    }
    
    __host__ __device__ double
    Option_pricer_montecarlo::Get_P2() const
    {
        return _P2;
    }

    __host__ __device__ size_t
    Option_pricer_montecarlo::Get_N() const
    {
        return _N;
    }

    __host__ __device__ void
    Option_pricer_montecarlo::Set_N(size_t N)
    {
        _N = N;
    }

    __host__ __device__ void
    Option_pricer_montecarlo::resimulate_option()
    {
        simulate_option();
    }


    __host__ __device__ void
    Option_pricer_montecarlo::simulate_option()
    {
        double p= 0.;
        double p_sqrd = 0.;


        //Contract_eq_option &contract = 
                //static_cast<Contract_eq_option&>(*_contract_option);
        Schedule * schedule = contract_option->Get_schedule();
        Equity_prices * starting_point = contract_option->Get_eq_prices();
        Path path(starting_point,schedule, _process);

        for(size_t i = 0; i < _N; ++i)
        {

            pay_off = contract.Pay_off(&path);
            if()
            if(pay_off)
        
            path.regen_path();
        }    
        
        _price = pay_off / double(_N);
        _price_square = pay_off2;  //cos'è la somma dei quadrati ---> forse meglio cambiargli nome

        //compute_MC_error();
    }

    /**
     * @brief compute the MC error according to the formula:
     * \sigma_{MC} = \sigma/\sqrt(N)         
     * \sigma = <f^2> - <f>^2 
     */
    __host__ __device__ void
    Option_pricer_montecarlo::compute_MC_error()
    {
        _MC_error =  (_price_square/static_cast<double>(_N) - _price*_price)/ //sigma 
                                                    sqrt(static_cast<double>(_N));
    }
}



