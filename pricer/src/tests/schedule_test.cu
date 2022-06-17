#include <iostream>
#include "../lib/support_lib/parse_lib/parse_lib.cuh"




int main(int argc, char ** argv)
{
    const size_t size = 10;
    double * schedule = new double[size];
    for(int i = 0; i < size; ++i)
        schedule[i] = (double)i/10.;

    double * dev_schedule;



    if(prcr::cmdOptionExists(argv,argc + argv, "-cpu"))
    {

    }
    else
    {
        int device_count = 10;
        if(cudaGetDeviceCount(&device_count) == 100)
        {
            std::cerr << "no cuda device foundend\n";
            return -1;
        }
        else
        {
            std::cout << device_count 
                      << " device avaible\n";
        }
        // delete device;
    }




}