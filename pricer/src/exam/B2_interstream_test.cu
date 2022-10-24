#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/support_lib/timer_lib/myTimer.cuh"
#include <cmath>


//Questo test verifica che l�algoritmo scelto per la generazione dei numeri pseudo-casuali multi-stream 
//produce delle sequenze inter-stream tra loro indipendenti.
//Come suggerito, funziona solo su GPU


#define THREADS 10000
#define STEPS 1000

__host__ bool gen_sequence(uint*, double*,int);
__host__ bool check_correlation(uint*, double*);



__host__ bool check_correlation(uint* seeds, double* corr_array)
    {
        double arr_1[STEPS];
        double arr_2[STEPS];
        bool status = true;
        
        

        status = status && gen_sequence(seeds, arr_1, 0);
        bool keep_arr_1 = true;
        bool keep_arr_2 = false;

    for (int j = 0; j < THREADS-1; j++)
    {
        if (!keep_arr_1)
        {
            status = status && gen_sequence(seeds, arr_1, j+1);
        }
            
        if (!keep_arr_2)
        {
            status = status && gen_sequence(seeds, arr_2, j+1);
        }

        corr_array[j] = 0;
        for (int i = 0; i < STEPS; i++)
        {
                corr_array[j] += arr_1[i] * arr_2[i];
        }

        keep_arr_1 = !keep_arr_1; 
        keep_arr_2 = !keep_arr_2;
    }

    return status;
}
__host__ bool gen_sequence(uint* seeds, double* arr, int index)
{
    uint seed0 = seeds[4 * index];
    uint seed1 = seeds[4 * index + 1];
    uint seed2 = seeds[4 * index + 2];
    uint seed3 = seeds[4 * index + 3];

    rnd::MyRandomImplementation* gnr = new rnd::GenCombined(seed0, seed1, seed2, seed3);        
    bool flag = true;
    for (size_t i = 0; i < STEPS; i++)
    {
        if (!gnr->Get_status()) { flag = false; }
        arr[i] = gnr->genGaussian(0.,1.);
    }
    delete(gnr);
    return flag;
}


int main(int argc, char** argv)
{
    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;

    double corr_array[THREADS - 1];
    bool status = true;

    srand(time(NULL));
    uint seed_aus[4];
    for (size_t i = 0; i < 4; i++)
    {
        seed_aus[i] = rnd::genSeed(true);
    }

    uint* seeds = new uint[4 *THREADS];
    rnd::GenCombined gnr_aus(seed_aus[0], seed_aus[1], seed_aus[2], seed_aus[3]);
    for (size_t i = 0; i < 4 * THREADS; i++)
    {
        seeds[i] = gnr_aus.genUniformInt();
        while (seeds[i] <= 128)
        {
            seeds[i] = gnr_aus.genUniformInt();
        }
    }


    if (dev.CPU)
    {
        int result = 0;
        status = check_correlation(seeds, corr_array);
        if (!status)
            printf("Errore nella creazione di un generatore\n");
        for (int j = 0; j < THREADS - 1; j++)
        {
            if (abs(corr_array[j]/STEPS) > 5 * 1. / sqrt(STEPS))
            {
                //std::cout << "Ci sono correlazioni tra gli stream: " << j << " e " << j + 1 << std::endl;
                //std::cout << "Correlazione: " << corr_array[j]/STEPS << std::endl << std::endl;
                result++;
            }
            else 
            {
                //std::cout << "nessuna correlazione\n";
            }
        }
        if (!status)
            printf("Errore nella creazione di un generatore\n");

        delete[](seeds);
        std::cout << "Numero di stream correlati: " << result << std::endl;
        return result;
    }



    if (dev.GPU)
    {
        std::cout << "This test only works on CPU" << std::endl;
        return 0;
    }

    
}
