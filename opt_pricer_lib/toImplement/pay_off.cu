#include "../include/pay_off.hpp"

namespace pricer
{
    HD PayOff_vanilla::PayOff_vanilla(double f_S, double E)
        :_f_S(f_S),_E(E)
    {
    }

//________________________________________________________________________________________________

    HD double PayOff_vanilla::getPayOff_call()
    {
        return max(_f_S - _E,0.);
    }
    HD double PayOff_vanilla::getPayOff_put()
    {
        return max(_E - _f_S,0.);
    }
//________________________________________________________________________________________________

    HD PayOff_digital::PayOff_digital(double f_S, double E, double K)
        :PayOff_vanilla(f_S,E),_K(K)
    {
    }

    HD double PayOff_digital::getPayOff_call()
    {
        if (max(_f_S - _E, 0.)  > 0)
            return _K;
        else 
         return 0.;   
    }

    HD double PayOff_digital::getPayOff_put()
    {
        if (max(_E - _f_S,0.) > 0)
            return _K;
        else 
            return 0.;
    }

//________________________________________________________________________________________________
    HD PayOff_esotic::PayOff_esotic(const std::vector<double> & path, double E)
        :_path(path),_E(E)
    {
    }

//________________________________________________________________________________________________
    HD double PayOff_asiatic::getPayOff_call()
    {
        if(_avg_not_computed)
        {
            _avg = average(_path);
            _avg_not_computed = false;
        }
        return max(_avg - _E, 0.);
    }

    HD double PayOff_asiatic::getPayOff_put()
    {
        if(_avg_not_computed)
        {
            _avg = average(_path);
            _avg_not_computed = false;
        }
        return max(_E - _avg, 0.);
    }
//________________________________________________________________________________________________
    
}