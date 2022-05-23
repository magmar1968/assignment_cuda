#include <iostream>
#include <cmath>  //sqrt
#include <cassert>

typedef struct                  //l'output del singolo thread potrebbe essere di questo tipo,
{                               //poi risultati di thread diversi vengono raggruppati in un unico vettore con cudaMalloc

    double simple_sum;          //somma dei risultati delle simulazioni sul singolo thread
    double quadratic_sum;       //somma dei quadrati
    int sim_per_thread;         //numero di simulazioni eseguite dal thread se il numero totale di simulazioni non fosse noto o per controllarlo
} thread_output;

typedef struct
{
    double mean;
    double sim_error;       //risultati della simulazione desiderati a schermo
} sim_output;




class Statistica
    {
    public:

        Statistica(int thread_number, thread_output* vettore_output);
        ~Statistica(){};
        double get_mean();
        double get_error();

    private:

        void accumula();
        void calcola();
        int N;                          //numero di thread istanziati
        int N_simulations;              //numero di simulazioni totale (a priori ignoto?o per controllo)
        thread_output accum_results;    //risultato della somma su tutto il vettore dei risultati dei thread
        thread_output* ptr_to;          //puntatore al vettore dei risultati dei thread
        sim_output risultati;           //risultati della simulazione
        //bool error_bool;                //la deviazione standard non deve essere negativa ecc...




    };



Statistica::Statistica(int thread_number, thread_output* vettore_output)
{
    //error_bool = false;
    ptr_to = vettore_output;
    N = thread_number;
    accumula();
    calcola();
}

void Statistica::accumula()  //accumula i risultati dei thread
{
    accum_results.simple_sum = 0;
    accum_results.quadratic_sum = 0;
    N_simulations = 0;
    for(int i = 0; i < N;i++)    //scorre sul vettore di risultati
    {
        accum_results.simple_sum += ptr_to[i].simple_sum;   //somme semplici
        assert(ptr_to[i].quadratic_sum>=0);
        accum_results.quadratic_sum += ptr_to[i].quadratic_sum;



        /*if(ptr_to[i].quadratic_sum>=0)
            {accum_results.quadratic_sum+=ptr_to[i].quadratic_sum;}   //somme dei quadrati
        else
        {
            error_bool = true;
            std::cout<<"Un thread ha prodotto una varianza negativa\n"<<std::endl;
        }*/

        N_simulations+=ptr_to[i].sim_per_thread;            //somma del numero di simulazioni eseguite dai thread
    }

}

void Statistica::calcola()
{
    risultati.mean = accum_results.simple_sum/(N_simulations);
    double var = accum_results.quadratic_sum/N-risultati.mean*risultati.mean;
    assert(var>=0);
    risultati.sim_error = sqrt(var);
}

double Statistica::get_mean()
{
    return risultati.mean;
}

double Statistica::get_error()
{
    //if(!error_bool)
    return risultati.sim_error;
    //else
    //return -1;
}




int main()
{


    thread_output a[100];       //ottenuto da cudaMalloc
    for (int i =0; i<100; i++)
    {
        a[i].simple_sum = i;
        a[i].quadratic_sum = i*i;
        a[i].sim_per_thread = 10;
    }

    Statistica stats(100,a);
    std::cout << "La media e': " << stats.get_mean() << std::endl;
    std::cout << "L'errore e': " << stats.get_error() << std::endl;
    return 0;
}
