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
    


    struct Device_options
    {
        bool   cpu;
        bool   gpu;
        size_t N_block; // # blocks
        size_t N_thread;// # threads

    };

    struct Yield_curve_args
    {
        bool   flat;       // true if flat yield curve
        double rate;       // rate for flat yield curve

        bool    structured;// true if structured yield curve
        double* rates;     // pointer to rates array
        double* times;     // pointer to times array
        size_t  dim;       // array dimension         
    };

    struct Volatility_args
    {
        double vol;    //volatility 
    };

    struct Schedule_args
    {
        double t_ref;
        double deltat;
        double dim;
    };
    
    struct Eq_descr_args
    {
        char   isin_code[10];
        char   name[15];
        char   currency[10];
        double dividend_yield;
    };
    
    struct Eq_price_args
    {
        double time;
        double price;
    };


    H bool ReadInputOption(std::string filename);








    struct Device
    {
        bool GPU = 0;
        bool CPU = 0;
    };





}


#endif 