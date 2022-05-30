#ifndef __LIB__
#define __LIB__
#include <math.h>
#include <vector>



namespace pricer
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    
    
    HD double average(const double * array, const size_t dim);

}


#endif 