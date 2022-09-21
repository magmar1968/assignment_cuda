#include <iostream>
#include "../lib/support_lib/parse_lib/parse_lib.cuh"


int main(int argc, char ** argv)
{
    if(prcr::cmdOptionExists(argv,argv + argc, "-gpu"))
    {
        std::cout << "hello gpu!\n";
    }
    else if(prcr::cmdOptionExists(argv,argv + argc, "-cpu"))
        std::cout << "hello cpu!\n";
}