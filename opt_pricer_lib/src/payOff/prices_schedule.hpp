
/*Definizione delle classi option prices e schedule (utili a path e pricer monte carlo) sulla falsariga delle dispense*/

#include <iostream>
#include <cassert>


namespace pricer
{
    /*#define H __host__
    #define D __device__
    #define HD __host__ __device__ */

    class Option_prices {           //la classe contiene i prezzi di più sottostanti a t fissato
        private:
            double time ;            //tempo al quale l'oggetto si riferisce
            double *prices ;         //pointer al vettore di prezzi
            int dim ;

        public:
            Option_prices(double time_init, double *prices_init, int dim);   //inizializza i prezzi dei sottostanti per t iniziale
            double Get_time(void);
            void Set_time(double time_init);
            double Get_price(int i);
            int Set_price(int i, double price_init);                         //setter per gli oggetti che si riferiscono a t successivi a quello iniziale
    };

   /*---------------------------------------------------------------------------------------------------------------*/


    class Schedule                //contiene i tempi a cui si valutano i prices per comporre il path
    {
        private:

            double* t ;    //pointer al vettore dei tempi (years fraction)
            int dim ;      //dimensione del vettore
            bool Check_order();   //metodo per controllare se gli istanti di tempo sono in ordine crescente

        public:

            Schedule(double t_ref, double delta_t, int dim_init); //costruttore con delta_t in input
            Schedule(double* t_init, int dim_init);               //costruttore con il vettore dei tempi in input
            double Get_t(int i) ;                                 //recupera l'istante di tempo i-esimo in years fractions
            int Get_dim(void) ;
    };




    /*class Path  //bozza
    {
        private:
            Option_prices *starting_point ;
            Option_prices **Option_prices_scenario ;  //puntatore a un array di Option Prices (cioè l'evoluzione dei prezzi)
            int length ;

        public:
            Path(Option_prices *starting_point_init, Schedule *schedule, Process_eq *process_eq);
            Option_prices *Get_starting_point(void);
            Option_prices *Get_Option_prices(int i);
            int Get_length(void);

    };*/



    /*class Option      //bozza
    {
        Schedule* ptr_schedule;
        Option_prices* ptr_prices;
        StochProcess* ptr_process;
        Path* ptr_path;

    };*/
}




