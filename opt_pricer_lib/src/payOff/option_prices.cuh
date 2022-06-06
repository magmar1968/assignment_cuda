
/*Class option prices definition*/

#include <iostream>
#include <cassert>


namespace pricer
{
    /*#define H __host__
    #define D __device__
    #define HD __host__ __device__ */

    class Option_prices {           //la classe contiene i prezzi di pi� sottostanti a t fissato
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

}




