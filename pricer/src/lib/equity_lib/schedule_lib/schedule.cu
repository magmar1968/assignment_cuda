#include "schedule.cuh"

namespace pricer
{


    HD Schedule::Schedule(double t_ref, double delta_t, int dim)
        :_dim(dim)
    {
        _t = new double[_dim];
        if(delta_t<0)
        {
            for(int i = 0; i < _dim; i++)
            {
                _t[i] = 0;
                _ascending = false;
            }
        }
        else
        {
            for (int i = 0; i < _dim; i++)
            {
                _t[i] = t_ref + delta_t * i;
            }
            _ascending = true;
        }
        
        
    }


    HD Schedule::Schedule(double* t_init, int dim)
        :_dim(dim)
    {

        _t = new double[dim];
        for (int i = 0; i < _dim; i++)
        {
            _t[i] = t_init[i];
        }
        if(!Check_order())
        {
            _ascending = true;
        }
        else
        {
            
            for (int i = 0; i < _dim; i++)
            {
                _t[i] = 0;
            }
            _ascending = false;
        }
    }

    HD void Schedule::Get_t(double* ptr)
    {
        for (int i = 0; i < _dim; i++)
        {
            ptr[i] = _t[i];
        }
    }

    HD int Schedule::Get_dim(void)
    {
        return _dim;
    }

    HD bool Schedule::Check_order()
    {
        for (int i = 1; i < _dim; i++)
        {
            if (_t[i] <= _t[i - 1]) { return false; }
        }
        return true;
    }

    HD bool Get_order()
    {
        return _ascending;
    }

}