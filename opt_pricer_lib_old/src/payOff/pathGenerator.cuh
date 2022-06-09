#ifndef __PATHGENERATOR__
#define __PATHGENERATOR__

#include "../support_lib/myRandom/myRandom.cuh"
#include "../stoch_proc/stoch.cuh"
#include "schedule/schedule.cuh"

namespace pricer
{
#define H __host__
#define D __device__
#define HD __host__ __device__   


	class Path
	{
	public:
		HD Path(rnd::MyRandom* gnr, StochProcess* stc, Schedule* cal, size_t steps);
		HD double* getPath()    const;
		HD size_t  getN_Steps() const;


	protected:
		double* _path; // implementare container adeguato
		size_t _steps;
		StochProcess* _stc;
		rnd::MyRandom* _gnr;
		Schedule* _cal;
	};


	class PathImp : public Path
	{
	public:
		HD PathImp(rnd::MyRandom* gnr, StochProcess* stc, Schedule* cal, size_t steps);

	private:
		HD void genPath();
	};
}





#endif