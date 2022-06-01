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

        HD double get_number();
        HD void set_number(double number);  //forse è inutile
        HD bool check_sign();
    private:
        double  _number;   //se set è inutile allora è const
    };
}

