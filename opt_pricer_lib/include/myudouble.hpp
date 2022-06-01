/*Classe per i prices, option prices must be >0*/

namespace pricer
{
    class myudouble
    {
    public:
        myudouble(double number);

        double get_number();
        void set_number(double number);
        bool check_sign();

    private:
        double _number;
    };
}

