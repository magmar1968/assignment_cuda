#include "myRandom.hpp"
#include <random>
#include <iostream>
#include <fstream>

int main()
{
    srand(time(NULL));
    uint seed1 = rnd::genSeed(true);
    uint seed2 = rnd::genSeed(true);
    uint seed3 = rnd::genSeed(true);
    uint seed4 = rnd::genSeed();

    std::cout << "seed1: " << seed1 << std::endl
              << "seed2: " << seed2 << std::endl
              << "seed3: " << seed3 << std::endl
              << "seed4: " << seed4 << std::endl;
    rnd::GenCombined gnr(seed1,seed2,seed3,seed4);

    if(! gnr.getStatus())
        return -1;
    std::ofstream ofs("../data/data_gauss_comb.dat", std::ofstream::out);
    for(int i = 0; i < 50000000; ++i)
    {
        ofs << gnr.genGaussian() << "\n";
    }


    

}



