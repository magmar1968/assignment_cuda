* Schedule implementation* /
#include "../include/schedule.hpp"

namespace pricer
{


    HD Schedule::Schedule(double t_ref, double delta_t, int dim)
        :_dim(dim)
    {
        _t = new double[_dim];
        for (int i = 0; i < _dim; i++)
        {
            _t[i] = t_ref + delta_t * i;
        }
        assert(Check_order());
    }
    HD Schedule::Schedule(double* t_init, int dim)
        :_dim(dim)
    {

        _t = new double[dim];
        for (int i = 0; i < _dim; i++)
        {
            _t[i] = t_init[i];
        }
        assert(Check_order());
    }
    HD double Schedule::Get_t(int i)
    {
        return _t[i];
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
}

