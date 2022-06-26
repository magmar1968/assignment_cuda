#ifndef __LIB__
#define __LIB__
#include <math.h>
#include <algorithm>
#include <string>
#include <sstream> //string stream
#include <fstream>
#include <iostream>
#include <vector>

#include "../lib_lib/lib.cuh" //countWords

namespace prcr
{
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    
    
   
    H bool cmdOptionExists(char** begin, char** end, const std::string& option);
    H std::string getCmdOption(char ** begin, char ** end, const std::string & option);

    H bool fileOptionExist(std::string fileName, 
                           std::string option,
                           std::string *_line = NULL);

    
    template<typename T>
    H void fileGetOptionValue(std::string fileName, 
                              std::string option,
                              T           *out_value)
    {
        std::string line;
        if(fileOptionExist(fileName,option,&line) == false)
        {
            std::cerr << "ERROR: option "<< option << " doesn't exist\n";
            exit(-1);
            
        }
        
        std::stringstream ss(line);
        std::string option_value;
        ss >> option_value;//option name
        ss >> *out_value;
    }    
    
    
    template<typename T> 
    H void fileGetOptionVectorVal(std::string  fileName,
                                 std::string   option,
                                 std::vector<T> *out_values)
    {    
        std::string line;
        if(fileOptionExist(fileName,option,&line) == false)
        {
            std::cerr << "ERROR: option "<< option << " doesn't exist\n";
            exit(-1);
        }

        std::stringstream ss(line);
        std::string option_name;
        ss >> option_name;// skip option name
        
        T value;
        while(ss >> value)
        {       
            out_values->push_back(value);
        }
    }
    
    
    struct Device
    {
        bool GPU = 0;
        bool CPU = 0;
    };


}


#endif 