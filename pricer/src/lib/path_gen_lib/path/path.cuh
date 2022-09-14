#ifndef __PATHGENERATOR__
#define __PATHGENERATOR__

#include "../process.cuh"
#include "../process_eq_imp/process_eq_lognormal.cuh"
#include "../../equity_lib/equity_prices.cuh"
#include "../../equity_lib/schedule_lib/schedule.cuh"


#define H __host__
#define D __device__
#define HD __host__ __device__  


class Path {
  private:
	Equity_prices  * _starting_point;
	Equity_prices* * _eq_prices_scenario;
	Schedule       * _schedule; 
	Process_eq_lognormal * _process_eq_lognormal;
	double         * _random_numbers_scenario;
	size_t           _dim;  //number of steps
	size_t           _start_ind;


	HD void gen_path();
  public:
	//constructors & destructors
	HD Path(void){};
	HD Path(Equity_prices* starting_point,
			Schedule     * schedule,
			Process_eq_lognormal   * process_eq);
	HD Path(Path* path);
	HD virtual ~Path(void);
	
	//getter & setters
	HD Equity_prices  * Get_starting_point(void) const;
	HD Equity_prices  * Get_equity_prices(size_t i) const; //forse togliere il puntatore???
	HD double           Get_random_numbers(size_t i) const;
	HD size_t Get_dim(void) const;
	HD size_t Get_start_ind() const;
	//functions
	HD Equity_prices * operator[](size_t i) const;
	HD void regen_path();
	// HD void destroy ();

};

#endif 
