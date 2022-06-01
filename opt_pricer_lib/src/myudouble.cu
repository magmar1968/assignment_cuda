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


    HD void myudouble::set_number(double number)
    {
        _number = number;
        assert(check_sign());  //prossimamente: usare exit e ritornare codice
    }


    HD double myudouble::get_number()
    {
        return _number;
    }


    HD bool myudouble::check_sign()
    {
        return _number>0 ? true : false;
    }
}


//Per qualche prova
/*#include <iostream>
int main()
{
    pricer::myudouble num(0.0004);
    std::cout<<num.get_number()<<std::endl;
}*/
