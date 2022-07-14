#include "path.cuh"

__host__ __device__
Path::Path(Equity_prices* starting_point,
	Schedule* schedule,
	Process_eq* process_eq,
	rnd::MyRandom* gnr,
	double* rns,
	Equity_prices* eps)
	:_starting_point(starting_point), _gnr(gnr), _eq_prices_scenario(eps), _random_numbers_scenario(rns)
{

	 _dim = schedule -> Get_dim(); // n schedule steps
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
	}

	double delta_t = schedule->Get_t(_start_ind) - start_t;   //first step, from starting_point
	_random_numbers_scenario[_start_ind] = _gnr->genGaussian();
	process_eq->Get_new_price(&_eq_prices_scenario[_start_ind], _starting_point, _random_numbers_scenario[_start_ind], delta_t);

	for (size_t j =  _start_ind + 1; j < _dim; j++)              //makes steps--->creates scenario
	{
		 delta_t = schedule->Get_t(j) - schedule->Get_t(j-1);
		 _random_numbers_scenario[j] = _gnr->genGaussian();  //crea numeri random e li memorizza  //fare come lo step 1 con il setter
		 process_eq->Get_new_price(&_eq_prices_scenario[j],&_eq_prices_scenario[j - 1], _random_numbers_scenario[j],delta_t); 
		
	}
	 //per il momento, se processo non esatto ci accontentiamo dell'approssimazione
}



__host__ __device__ Equity_prices* 
Path::Get_starting_point(void) const
{
	return _starting_point;
}

__host__ __device__ Equity_prices 
Path::Get_equity_prices(size_t i) const
{
	return _eq_prices_scenario[i];
}

__host__ __device__ Random_numbers 
Path::Get_random_numbers(size_t i) const
{
	return _random_numbers_scenario[i];
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


/*__host__ __device__ Equity_prices*
Path::operator[](size_t i)const
{
	return Get_equity_prices(i);
}*/


__host__ __device__ void
Path::regen_path(Schedule  * schedule,
					   Process_eq* process_eq)
{
	gen_path(schedule,process_eq);
}