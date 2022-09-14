/*myudouble source file*/
#include <cassert>
#include "myudouble.cuh"
namespace prcr
{

    HD myudouble::myudouble(double number)
    :_number(number)
    {

        if (!check_sign())
        {
            _number = 0.;
        }
    }

    HD double myudouble::get_number() const
    {
        return _number;
    }


    HD bool myudouble::check_sign() const
    {
        return _number>0 ? true : false;
    }
    
    HD void myudouble::operator=(const myudouble& rhs)
    {
        this->_number = rhs.get_number();
    }
    
    HD void myudouble::operator=(const double& rhs)
    {
        if(myudouble(rhs).check_sign())
            _number = rhs;
        else 
            _number = 0.;
    }


    HD myudouble& myudouble::operator+=(const myudouble& rhs)
    {
        _number += rhs.get_number();
        if(check_sign())
            return *this;
        else
        {
            _number = 0.0;
            return *this;
        }    
    }

    HD myudouble& myudouble::operator*=(const myudouble& rhs)
    {
        _number *= rhs.get_number();
        if (check_sign())
            return *this;
        else
        {
            _number = 0.0;
            return *this;
        }
    }
}


