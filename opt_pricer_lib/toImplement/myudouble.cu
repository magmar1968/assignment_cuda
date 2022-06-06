/*myudouble source file*/
#include <cassert>
#include "../include/myudouble.hpp"
namespace pricer
{
    HD myudouble::myudouble(double number)
    :_number(number)
    {

        assert(check_sign());    //prossimamente: usare exit e ritornare codice
    }


    /*HD void myudouble::set_number(double number)
    {
        _number = number;
        assert(check_sign());  //prossimamente: usare exit e ritornare codice      //utile?
    }*/


    HD double myudouble::get_number() const
    {
        return _number;
    }


    HD bool myudouble::check_sign() const
    {
        return _number>0 ? true : false;
    }
}


