#include "myRandom.h"
#include <random>
#include <iostream>
#include <fstream>

int main()
{
    rnd::GenTausworth gnr(200,TAUSWORTH_2);

    gnr.setGaussImpl(GAUSSIAN_2);

    std::ofstream ofs("../data/data_unif_taus2.dat", std::ofstream::out);
    for(int i = 0; i < 50000000; ++i)
    {
        ofs << gnr.genUniform() << "\n";
    }

}



