/*Definizione delle classe schedule*/
#include <iostream>
#include <cassert>


namespace pricer
{
#define H __host__
#define D __device__
#define HD __host__ __device__


    class Schedule                //contiene i tempi a cui si valutano i prices per comporre il path
    {
    private:

        //pointer al vettore dei tempi (years fraction)
        int _dim;  //dimensione del vettore
        double* _t;

        HD bool Check_order();   //metodo per controllare se gli istanti di tempo sono in ordine crescente
        //HD bool Check_sign(); //accettiamo tempi negativi?

    public:

        HD Schedule(double t_ref, double delta_t, int dim); //costruisce un vettore i cui elementi sono equidistanti
        HD Schedule(double* t_init, int dim);               //costruttore con il vettore dei tempi in input
        HD double Get_t(int i);                                 //recupera l'istante di tempo i-esimo in years fractions
        HD int Get_dim(void);

    };
}
