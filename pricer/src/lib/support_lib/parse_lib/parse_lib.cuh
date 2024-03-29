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
    
    

    H inline bool file_exists (const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }
   
    H bool cmdOptionExists(char** begin, char** end, const std::string& option);
    H std::string getCmdOption(char ** begin, char ** end, const std::string & option);

    H bool fileOptionExist(std::string fileName, 
                           std::string option,
                           std::string *_line = NULL);

    
    template<typename T>
    H bool fileGetOptionValue(std::string fileName, 
                              std::string option,
                              T           *out_value)
    {
        std::string line;
        if(fileOptionExist(fileName,option,&line) == false)
        {
            std::cerr << "ERROR: option "<< option << " doesn't exist\n";
            return false;
            
        }
        
        std::stringstream ss(line);
        std::string option_value;
        ss >> option_value;//option name
        ss >> *out_value;
        return true; 
    }    
    
    
    template<typename T> 
    H bool fileGetOptionVectorVal(std::string  fileName,
                                 std::string   option,
                                 std::vector<T> *out_values)
    {    
        std::string line;
        if(fileOptionExist(fileName,option,&line) == false)
        {
            std::cerr << "ERROR: option "<< option << " doesn't exist\n";
            return false;
        }

        std::stringstream ss(line);
        std::string option_name;
        ss >> option_name;// skip option name
        
        T value;
        while(ss >> value)
        {
            out_values->push_back(value);
        }
        return true;
    }
    


    struct Dev_opts
    {
        bool   CPU;
        bool   GPU;
        size_t N_blocks; // # blocks
        size_t N_threads;// # threads

    };

    struct MC_args
    {
        size_t N_simulations;
    };


    struct Schedule_args
    {
        double T;
        double t_ref;   //initial time
        double deltat;  //time step size
        size_t dim;     //time steps
    };

    struct Contract_args
    {
        char   option_type[16]; //vanilla ecc.       
        char   contract_type;   //call or put 
        
        double strike_price;

        //esotic corridor contract arguments
        double B;   
        double N;
        double K;
        //add more stuff for different type of contracts 
    };
    
    struct Eq_descr_args
    {
        double rate;
        double vol;
        double dividend_yield;
    };
    
    struct Eq_price_args
    {
        double time;
        double price;
    };

    struct Device
    {
        bool GPU;
        bool CPU;
    };

    struct Stc_pr_args
    {
        bool exact;
        size_t l;
    };

    struct Pricer_args
    {
        Dev_opts      dev_opts;
        MC_args       mc_args;
        Contract_args contract_args;
        Schedule_args schedule_args;
        Eq_descr_args eq_descr_args;
        Eq_price_args eq_price_args;
        Stc_pr_args   stc_pr_args;
    };
    
    


    H bool ReadInputOption(std::string filename, 
                           Pricer_args * pricer_args);






}


#endif 