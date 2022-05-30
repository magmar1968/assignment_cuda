#include "../include/lib.hpp"

namespace pricer
{
    HD double average(const double * array, const size_t dim)
    {
        double avg = 0.;
        for( size_t it = 0; it < dim; it++)
        {
            avg += array[it]/ static_cast<double>(dim);
        }
        return avg; 
    }
    
     

}