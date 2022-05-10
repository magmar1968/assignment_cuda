#include "myRandom.h"
#include <iostream>

int main()
{
    rnd::LinGenCongruential<float> genRnd(7);

    for(int i = 10; i < 100 ; ++i)
        std::cout << genRnd.genUniform();


    return 0;
}



