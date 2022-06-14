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
	printf("start t: %f\n", start_t);
	//check for starting point in the schedule
	for(size_t k = 0; k < _dim; k++)
	{
		printf("schedule gett %d: %f\n", k, schedule->Get_t(k));
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

	printf("start_index: %d\n", _start_ind);

	double delta_t = schedule->Get_t(_start_ind) - start_t;   //first step, from starting_point
	Random_numbers* temp_random_array = process_eq->Get_random_structure();  //array di random numbers
	_random_numbers_scenario[_start_ind] = new Random_numbers(_n_eq);      //allocazione per rns
	for (size_t i = 0; i < _n_eq; i++)
	{
		_random_numbers_scenario[_start_ind]->Set_element(i, temp_random_array->Get_element(i)); //usa il vettore temp per inizializzare la prima colonna di rns
		printf("done\t");
	}
	//printf("delta t: %f\n", delta_t);
	//printf("dim starting point: %d\n", _starting_point->Get_dim());
	_eq_prices_scenario[_start_ind] = new Equity_prices(); 
	_eq_prices_scenario[_start_ind]= //usare setters e getters                                                 //bug to be fixed here
	    process_eq -> Get_new_prices(_starting_point, _random_numbers_scenario[_start_ind], delta_t);  


	for (size_t j =  _start_ind + 1; j < _dim; j++)              //makes steps--->creates scenario
	{
		 delta_t = schedule->Get_t(j+1) - schedule->Get_t(j);
		 _random_numbers_scenario[j] = process_eq->Get_random_structure(); //crea numeri random e li memorizza  //fare come lo step 1 con il setter
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
		exit(1);// probably it doesn't work on cuda
}

__host__ __device__ Random_numbers* 
Path::Get_random_numbers(size_t i) const
{
	if(i < _dim)
		return _random_numbers_scenario[i];
	else 
		exit(1);
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