#ifndef __PATHGENERATOR__
#define __PATHGENERATOR__

#include "myRandom.hpp"
#include "stoch.hpp"

namespace pricer
{
	#define H __host__
	#define D __device__
	#define HD __host__ __device__   


	class Path
	{
	public:
		HD Path();
		HD virtual double* getPath();
		HD virtual size_t getN_Steps();


	protected:
		double* _path;
		size_t _steps;
		rnd::MyRandom* _gnr;
		StochProcess* _stc;
		HD virtual void genPath();
	
	};
	

	class PathImp : public Path
	{
	public:
		HD PathImp(rnd::MyRandom* gnr, StochProcess* stc, size_t steps);
		HD double* getPath();
		HD size_t getN_Steps();

	private:
		HD void genPath();
	};






}





#endif