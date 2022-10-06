#include "path.cuh"

namespace prcr
{

	__host__ __device__
	Path::Path(Equity_prices * starting_point,
			   Schedule      * schedule,
			   Process_eq_lognormal   * process_eq)
		:_starting_point(starting_point), _schedule(schedule), _process_eq_lognormal(process_eq)
	{
		
		_dim = _schedule -> Get_dim(); // n schedule steps
		_eq_prices_scenario = new double[_dim];

		gen_path();
	}

	__host__ __device__
	Path::Path(Path* path)
	{
		//to do
		//se servisse, implementare funzione in class Random_numbers per creare l'opposto del vettore

	}

	__host__ __device__ Path::~Path()
	{
		delete[](_eq_prices_scenario);
	}

	__host__ __device__ void
	Path::gen_path()
	{
		double start_t = _starting_point -> Get_time();	
		double random_number;

		//check for starting point in the schedule
		//maybe better in another function
		for(size_t k = 0; k < _dim; k++)
		{
			if(start_t <= _schedule->Get_t(k))
			{
				_start_ind = k;
				break;                                                     //tutto questo pezzo va discusso meglio
			}
		}
		
		double delta_t = _schedule->Get_t(_start_ind) - start_t;   //first step, from starting_point
		random_number = _process_eq_lognormal->Get_random_gaussian();
		_eq_prices_scenario[_start_ind] = 
		                _process_eq_lognormal->Get_new_eq_price(_starting_point->Get_eq_description(), 
		                                                        _starting_point->Get_price(), 
																random_number, 
																delta_t);

		for (size_t j =  _start_ind + 1; j < _dim; j++)
		{
			delta_t = _schedule->Get_t(j) - _schedule->Get_t(j-1);

			random_number = _process_eq_lognormal->Get_random_gaussian(); 
			_eq_prices_scenario[j] = 
				_process_eq_lognormal->Get_new_eq_price(_starting_point->Get_eq_description(),
				                                        _eq_prices_scenario[j - 1],
														random_number,
														delta_t); 
			
		}
	}



	__host__ __device__ Equity_prices *
	Path::Get_starting_point(void) const
	{
		return _starting_point;
	}

	__host__ __device__ double
	Path::Get_equity_prices(size_t i) const
	{
		if(i < _dim)
			return _eq_prices_scenario[i];
		else 
			return -100; //exit(1);// probably it doesn't work on cuda
	}

	__host__ __device__ double 
	Path::Get_last_eq_price() const
	{
		return _eq_prices_scenario[_dim -1];
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


	__host__ __device__ double  
	Path::operator[](size_t i)const
	{
		return Get_equity_prices(i);
	}



	__host__ __device__ void 
	Path::regen_path()
	{
		gen_path();
	}

}
