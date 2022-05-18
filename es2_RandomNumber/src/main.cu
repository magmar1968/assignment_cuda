#include "myRandom.h"
#include <random>
#include <iostream>

int main()
{

    srand(time(NULL));
    uint iterations = 2000000;
    rnd::GenCombined gnr(rand(),rand(),rand(),rand());
    double somma = 0;
    for(int i =0; i < iterations ; i++)
    {
        somma += gnr.genUniform(3.,5.) ;
    }
    double fraz = somma/ static_cast<double> (iterations);
    std::cout << fraz << std::endl;
    return 0;
}



