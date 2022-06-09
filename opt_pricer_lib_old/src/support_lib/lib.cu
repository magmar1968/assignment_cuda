#include "./lib.cuh"

namespace pricer
{
    struct Device
    {
        bool GPU = 0;
        bool CPU = 0;
    };
    
    HD double average(const double * array, const size_t dim)
    {
        double avg = 0.;
        for( size_t it = 0; it < dim; it++)
        {
            avg += array[it]/ static_cast<double>(dim);
        }
        return avg; 
    }

    H bool cmdOptionExists(char** begin, char** end, const std::string& option)
    {
        return std::find(begin, end, option) != end;
    }

    H std::string getCmdOption(char ** begin, char ** end, const std::string & option)
    {
        char ** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return *itr;
        }
        return 0;
    }

    

}