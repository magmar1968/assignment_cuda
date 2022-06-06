#ifndef __LIB__
#define __LIB__
#include <math.h>
#include <algorithm>
#include <string>


namespace pricer
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    
    
    HD double average(const double * array, const size_t dim);
    H bool cmdOptionExists(char** begin, char** end, const std::string& option);
    H std::string getCmdOption(char ** begin, char ** end, const std::string & option);


    struct Device
    {
        bool GPU = 0;
        bool CPU = 0;
    };


}


#endif 