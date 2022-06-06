#ifndef __MYUDOUBLE__
#define __MYUDOUBLE__

/*Classe per i prices, option prices must be >0*/
namespace pricer
{


#define H __host__
#define D __device__
#define HD __host__ __device__   



    class myudouble
    {
    public:
        HD myudouble(double number);

        HD double get_number() const;
        //HD void set_number(double number); //utile?
        HD bool check_sign() const;

    private:
        double  _number;
       
    };

}

typedef pricer::myudouble udb;

#endif