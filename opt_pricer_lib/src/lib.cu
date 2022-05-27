#include "../include/lib.hpp"

namespace pricer
{
    HD double average(const std::vector<double> & array)
    {
        double avg = 0.;
        size_t dim = array.size();
        for( size_t it = 0; it < dim; it++)
        {
            avg += array[it]/ static_cast<double>(dim);
        }
        return avg; 
    } 

}