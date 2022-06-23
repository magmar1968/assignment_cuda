#ifndef __LIB__
#define __LIB__
#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>


namespace prcr
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    
    
   
    H bool cmdOptionExists(char** begin, char** end, const std::string& option);
    H std::string getCmdOption(char ** begin, char ** end, const std::string & option);

    H std::string preprocessInputFile(const std::string filename = "pricer.input");
    H int pricerInput(const std::string& inputFile,const std::string option);

    struct Device
    {
        bool GPU = 0;
        bool CPU = 0;
    };


}


#endif 