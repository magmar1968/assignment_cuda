#include "pathGenerator.cuh"


HD Path(Equity_prices* starting_point,
		Schedule* schedule,
		Process_eq* process_eq)
{
	// definire dei membri della classe di tipo dato per memorizzare schedule e process_eq dal costruttore?

	_dim = starting_point.Get_dim();   //numero delle diverse azioni che consideriamo nella simulazione
	size_t path_len = schedule.Get_dim();    //number of steps
	for (size_t i = 0; i < _dim; i++)
	{
		_starting_point[i] = starting_point[i];   
	}  


	double starting_time = starting_point->_time;
	for (size_t k = 0; k < path_len; k++)                 //looks for the starting point in schedule
	{
		if (starting_time == schedule->_t[k])             //<--to do: funzione per il confronto? 
		{
			break;
		}
	}   //ora k è l'index dello starting point in schedule


	double delta_t = schedule->_t[k] - starting_time;                            //first step, from starting_point
	_random_numbers_scenario[0] = process_eq.Get_random_structure(); //crea colonna di rnd_numbers, lunghezza = 1 se process non è multivariate
	_equity_prices_scenario[i] = process_eq.Get_new_prices(starting_point, rnd_num[k], delta_t);  

	for (size_t j = k; j < path_len; j++)              //makes steps--->creates scenario
	{
		 delta_t = schedule->_t[j + 1] - schedule->_t[j];
		 _random_numbers_scenario[j] = process_eq.Get_random_structure(); //crea numeri random e li memorizza
		 _equity_prices_scenario[j] = process_eq.Get_new_prices(_equity_prices_scenario[j - 1], rnd_num[j], delta_t); 
	}
	//per il momento, se processo non esatto ci accontentiamo dell'approssimazione


	
}


HD Path(Path* path)
{
	//to do
	//se servisse, implementare funzione in class Random_numbers per creare l'opposto del vettore

}

HD Equity_prices* 
Get_starting_point(void) const
{
	return _starting_point;
}

HD Equity_prices* 
Get_equity_prices(int i) const
{
	return _eq_prices_scenario[i];
}

HD Random_numbers* 
Get_random_numbers(int i) const
{

	return _random_numbers_scenario[i];
}

HD size_t
Get_dim(void) const
{
	return _dim;
}