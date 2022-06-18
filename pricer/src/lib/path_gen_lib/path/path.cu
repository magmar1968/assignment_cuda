#include "path.cuh"

__host__ __device__
Path::Path(Equity_prices* starting_point,
		   Schedule     * schedule,
		   Process_eq   * process_eq)
	:_starting_point(starting_point)
{

	 _dim = schedule -> Get_dim(); // n schedule steps
	 _n_eq = _starting_point -> Get_dim(); //n equities   //se serve, da qualche parte, fare check che number of equities sia coerente in tutti gli oggetti 
	 _random_numbers_scenario = new Random_numbers* [_dim];
	 _eq_prices_scenario = new Equity_prices * [_dim];
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
		if(start_t <= schedule->Get_t(k))
		{
			_start_ind = k;
			break;                                                     //tutto questo pezzo va discusso meglio
		}
		/*else if (start_t > schedule->Get_t(k))
		{
			_start_ind = k - 1;         
			break;
		}*/
	}

	double delta_t = schedule->Get_t(_start_ind) - start_t;   //first step, from starting_point
	//_random_numbers_scenario[_start_ind] = NULL;
	_random_numbers_scenario[_start_ind] = process_eq->Get_random_structure();
 
	//_eq_prices_scenario[_start_ind] = NULL;
	_eq_prices_scenario[_start_ind] = process_eq->Get_new_prices(_starting_point, _random_numbers_scenario[_start_ind], delta_t);

	for (size_t j =  _start_ind + 1; j < _dim; j++)              //makes steps--->creates scenario
	{
		 delta_t = schedule->Get_t(j) - schedule->Get_t(j-1);

		// _random_numbers_scenario[j] = NULL;
		 _random_numbers_scenario[j] = process_eq->Get_random_structure(); //crea numeri random e li memorizza  //fare come lo step 1 con il setter


		 _eq_prices_scenario[j] = NULL;
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
Path::Get_equity_prices(size_t i) const
{
	if(i < _dim)
		return _eq_prices_scenario[i];
	else 
		return NULL; //exit(1);// probably it doesn't work on cuda
}

__host__ __device__ Random_numbers* 
Path::Get_random_numbers(size_t i) const
{
	if(i < _dim)
		return _random_numbers_scenario[i];
	else 
		return NULL; //exit(1);
}

__host__ __device__ size_t
Path::Get_dim(void) const
{
	return _dim;
}

__host__ __device__ size_t
Path::Get_start_ind() const
{
	return _start_ind;
}


__host__ __device__ Equity_prices * 
Path::operator[](size_t i)const
{
	return Get_equity_prices(i);
}