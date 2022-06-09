#include "pathGenerator.cuh"
#include "cmath"
namespace pricer
{
	HD Path::Path(rnd::MyRandom* gnr, StochProcess* stc, Schedule* cal, size_t steps)
		:_gnr(gnr), _stc(stc), _cal(cal), _steps(steps)
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

	HD PathImp::PathImp(rnd::MyRandom* gnr, StochProcess* stc, Schedule* cal, size_t steps)
		:Path(gnr, stc, cal, steps)
	{
		genPath();
	}


	HD void PathImp::genPath()
	{
		/*double* rnd_array = new double[_steps];
		for (int i = 0; i < _steps; ++i)                  //questa parte non serve se non ci interessa memorizzare i numeri estratti
			rnd_array[i] = _gnr->genGaussian();*/


		_path[0] = _stc->getS();
		/*for (int it = 1; it < _steps; ++it)
		{
			_path[it] = _stc->get_step(_gnr->genGaussian());      //questa non serve se c'� schedule
		}*/


		double dt = _stc->get_dt();    //adesso prende dt dal procstoc
		int prox = 1;
		double temp;
		double *date =  new double [_steps];  //prende le date dalla schedule
		_cal->Get_t(date);
		if (_stc->get_exact())
		{
			//bisogna settare dt del processo stocastico, dipenderà da come è fatta la schedule. friend class per cambiare dt a procstoc?
		}
		else
		{
			//controlla di essere entro la data finale della schedule
			for (double tme = date[0];
				tme < date[_steps - 1] || abs(tme - date[_steps - 1]) < 0.001/*???*/;      
				tme += dt)
			{
				temp = _stc->get_step(_gnr->genGaussian());
				if (abs(date[prox] - tme) < 0.0005)                   //controlla se il tempo corrisponde a uno dei time nella schedule
				{
					_path[prox] = temp;
					prox++;
				}
			}
		}
	}
}