/*myudouble source file*/
#include <cassert>
#include "myudouble.cuh"
namespace pricer
{

    HD myudouble::myudouble(double number)
    :_number(number)
    {

        assert(check_sign());    //prossimamente: usare exit e ritornare codice
    }

    HD double myudouble::get_number() const
    {
        return _number;
    }


    HD bool myudouble::check_sign() const
    {
        return _number>0 ? true : false;
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
}


