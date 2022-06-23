#ifndef __PATHGENERATOR__
#define __PATHGENERATOR__

#include "../process.cuh"
#include "../process_eq.cuh"
#include "../../equity_lib/equity_prices.cuh"
#include "../../support_lib/myRandom/random_numbers.cuh"
#include "../../equity_lib/schedule_lib/schedule.cuh"


#define H __host__
#define D __device__
#define HD __host__ __device__  


class Path {
private:
	Equity_prices*   _starting_point;
	Equity_prices ** _eq_prices_scenario;
	Random_numbers** _random_numbers_scenario;
	size_t           _dim;  //number of steps
	size_t           _start_ind;
	size_t           _n_eq; //number of equities


	HD void gen_path(Schedule  * schedule,
					 Process_eq* process_eq);
public:
	//constructors & destructors
	HD Path(void){};
	HD Path(Equity_prices* starting_point,
			Schedule     * schedule,
			Process_eq   * process_eq);
	HD Path(Path* path);
	HD virtual ~Path(void){};
	//getter & setters
	HD Equity_prices  * Get_starting_point(void) const;
	HD Equity_prices  * Get_equity_prices(size_t i) const;
	HD Random_numbers * Get_random_numbers(size_t i) const;
	HD size_t Get_dim(void) const;
	HD size_t Get_start_ind() const;
	//functions
	HD Equity_prices * operator[](size_t i) const;
	HD void regen_path(Schedule  * schedule,
					   Process_eq* process_eq);

};

#endif 