#include "../include/pathGenerator.hpp"

namespace pricer
{
	HD PathImp::PathImp(rnd::MyRandom* gnr, StochProcess* stc, size_t steps)
	:_gnr(gnr),_stc(stc),_steps(steps)
	{
		_path = new double[_steps];
		genPath();
	}


	HD void PathImp::genPath()
	{
		//double rnd_array[_steps];
		double *rnd_array = _gnr->genGaussianVector(_steps);
		_path[0] = _stc->getS();
		for(int it = 1; it < _steps; ++it)
		{
			_path[it] = _stc->get_step(rnd_array[it]);
		}
	}

}