#include "pathGenerator.cuh"

namespace pricer
{
	HD Path::Path(rnd::MyRandom* gnr, StochProcess* stc, size_t steps)
		:_gnr(gnr), _stc(stc), _steps(steps)
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
		:Path(gnr, stc, steps)
	{
		genPath();
	}


	HD void PathImp::genPath()
	{
		//double rnd_array[_steps];
		/*double* rnd_array = new double[_steps];
		for (int i = 0; i < _steps; ++i)                      //questa parte non serve forse
			rnd_array[i] = _gnr->genGaussian();*/
		_path[0] = _stc->getS();
		/*for (int it = 1; it < _steps; ++it)
		{
			_path[it] = _stc->get_step(_gnr->genGaussian());      //questa serve se non c'� schedule
		}*/

		double t_init = _cal->Get_t(0);
		double t_final = _cal->Get_t(_steps - 1);
		double dt = _stc->get_dt();    //adesso prende dt dal procstoc
		int prox = 1;

		for (double tm = t_init; tm <= t_final; tm += dt)
		{

			if (tm == (_cal->Get_t(prox) - dt))
			{
				_path[prox] = _stc->get_step(_gnr->genGaussian());
				prox++;
			}
			else
			{
				_stc->get_step(_gnr->genGaussian());
			}

		}

	}

}