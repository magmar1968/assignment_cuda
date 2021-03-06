#include "option_pricer_montecarlo.cuh"

__host__ __device__
Option_pricer_montecarlo::Option_pricer_montecarlo(
                Contract_option *contract_option,
                Process         *process,
                size_t             N)
        :Option_pricer(contract_option,process),_N(N)
{
    simulate_option();
}

__host__ __device__ double 
Option_pricer_montecarlo::Get_price() const
{
    return _price;
}

__host__ __device__ double 
Option_pricer_montecarlo::Get_MonteCarlo_error() const
{
    return _error;
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
    double * pay_off  = new double[_N];
    double * pay_off2 = new double[_N];


    Contract_eq_option &contract = 
               static_cast<Contract_eq_option&>(*_contract_option);
    Schedule * schedule = contract.Get_schedule();
    Equity_prices * starting_point = contract.Get_eq_prices();
    Path  path =  Path(starting_point,schedule,
                        &static_cast<Process_eq&>(*_process));

    for(size_t i = 0; i < _N; ++i)
    {

        pay_off[i] = contract.Pay_off(&path);
        pay_off2[i] = pay_off[i]*pay_off[i];
	path.destroy();
        path.regen_path(schedule,&static_cast<Process_eq&>(*_process));
    }    

    
    _price = prcr::avg(pay_off,_N);
    _error = prcr::sum_array(pay_off2,_N);  //cos? ? la somma dei quadrati ---> forse meglio cambiargli nome

    delete[](pay_off);delete[](pay_off2);
}



