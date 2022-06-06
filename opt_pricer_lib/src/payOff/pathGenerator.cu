#include "../include/pathGenerator.hpp"

namespace pricer
{
	HD Path::Path(rnd::MyRandom* gnr, StochProcess* stc, size_t steps)
		:_gnr(gnr),_stc(stc),_steps(steps)
	{
		_path = new double[_steps];
	}

	HD double* Path::getPath() const
	{
		return _path;
	}

	HD size_t Path::getN_Steps() const
	{
		return _steps;
	}

	HD PathImp::PathImp(rnd::MyRandom* gnr, StochProcess* stc, size_t steps)
		:Path(gnr,stc,steps)
	{
		genPath();
	}


	HD void PathImp::genPath()
	{
		//double rnd_array[_steps];
		double *rnd_array = new double[_steps];
		for(int i = 0; i < _steps; ++i)
			rnd_array[i] = _gnr->genGaussian();
		_path[0] = _stc->getS();
		for(int it = 1; it < _steps; ++it)
		{
			_path[it] = _stc->get_step(rnd_array[it]);
		}
	}

}