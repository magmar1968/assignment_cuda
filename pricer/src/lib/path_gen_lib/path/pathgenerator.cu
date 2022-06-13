#include "pathgenerator.cuh"

__host__ __device__
Path::Path(Equity_prices* starting_point,
		   Schedule     * schedule,
		   Process_eq   * process_eq)
	:_starting_point(starting_point)
{

	_dim = schedule -> Get_dim(); // n schedule steps
	size_t n_eq = _starting_point -> Get_dim(); //n equities

	gen_path(schedule, process_eq);
}

__host__ __device__
Path::Path(Path* path)
{
	//to do
	//se servisse, implementare funzione in class Random_numbers per creare l'opposto del vettore

}

__host__ __device__ void
Path::gen_path(Schedule * schedule,
               Process_eq * process_eq)
{
	double start_t = _starting_point -> Get_time();	
	//check for starting point in the schedule
	for(size_t k = 0; k < _dim; k++)
	{
		if(start_t == schedule->Get_t(k))
		{
			_start_ind = k;
			break;
		}
		else if(start_t > schedule->Get_t(k))
		{
			_start_ind = k - 1;
			break;
		}
	}
	
	double delta_t = schedule->Get_t(_start_ind + 1) - schedule->Get_t(_start_ind);                            //first step, from starting_point
	_random_numbers_scenario[_start_ind]=
	                       process_eq ->Get_random_strucure(); //crea colonna di rnd_numbers, lunghezza = 1 se process non � multivariate
	_eq_prices_scenario[_start_ind]=
	    process_eq -> Get_new_prices(_starting_point, _random_numbers_scenario[_start_ind], delta_t);  



	for (size_t j =  _start_ind + 1; j < _dim; j++)              //makes steps--->creates scenario
	{
		 delta_t = schedule->Get_t(j+1) - schedule->Get_t(j);
		 _random_numbers_scenario[j] = process_eq->Get_random_strucure(); //crea numeri random e li memorizza
		 _eq_prices_scenario[j] = 
		       process_eq->Get_new_prices(_eq_prices_scenario[j - 1], _random_numbers_scenario[j],delta_t); 
	}
	// //per il momento, se processo non esatto ci accontentiamo dell'approssimazione
}


	


__host__ __device__ Equity_prices* 
Path::Get_starting_point(void) const
{
	return _starting_point;
}

__host__ __device__ Equity_prices* 
Path::Get_equity_prices(int i) const
{
	return _eq_prices_scenario[i];
}

__host__ __device__ Random_numbers* 
Path::Get_random_numbers(int i) const
{

	return _random_numbers_scenario[i];
}

__host__ __device__ size_t
Path::Get_dim(void) const
{
	return _dim;
}