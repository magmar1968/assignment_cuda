/*Schedule implementation*/
#include "../include/schedule.hpp"

namespace pricer
{


    HD Schedule::Schedule(double t_ref, double delta_t, int dim_init)
    {
        dim = dim_init;
        t = new double[dim];
        for(int i=0;i<dim;i++)
        {
            t[i] = t_ref +delta_t * i;
        }
        assert(Check_order());
    }
    HD Schedule::Schedule(double *t_init, int dim_init)
    {
        dim = dim_init;
        t = new double[dim];
        for(int i=0;i<dim;i++)
        {
            t[i] = t_init[i];
        }
        assert(Check_order());
    }
    HD double Schedule::Get_t(int i)
    {
        return t[i];
    }
    HD int Schedule::Get_dim(void)
    {
        return dim;
    }
    HD bool Schedule::Check_order()
        {
            for(int i=1;i<dim;i++)
            {
                if(t[i]<=t[i-1]) {return false;}
            }
            return true;
        }
}

