#include <iostream>
#include <string>
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include <fstream>

int main()
{
    std::string line;
    std::string fileName = "./data/example_file.txt";
    if(prcr::fileOptionExist(fileName,"#test1", &line) == false)
    {
        std::cerr << "ERROR: option not founded\n";
        return -1;
    }
    
    int         int_val;
    double      d_val;
    std::string s_val;
    bool        b_val;
    prcr::fileGetOptionValue(fileName, "#test_int", &int_val);
    if(int_val != 10)
        return 1;
    prcr::fileGetOptionValue(fileName, "#test_bool", &b_val);
    if(b_val != true)
        return 2;
    prcr::fileGetOptionValue(fileName, "#test_double", &d_val);
    if(d_val != (double)1.0333)
        return 3;
    prcr::fileGetOptionValue(fileName, "#test_string", &s_val);
    if(s_val != std::string("ciao?"))
        return 4;
    

    std::vector<std::string> s_values;
    prcr::fileGetOptionVectorVal(fileName, "#test_s_array", &s_values);
    
    std::vector<std::string> check_val{"ciao","come","stai?"};
    for(size_t i = 0; i < s_values.size(); ++i )
        if(s_values[i] != check_val[i]) return 5;

    std::vector<double> d_values;
    prcr::fileGetOptionVectorVal(fileName, "#test_d_array", &d_values);
    std::vector<double> d_check_val{3.14, 7.8111, 4111.};
    for(size_t i = 0; i < d_values.size(); ++i )
        if(d_values[i] != d_check_val[i]) return 6;
    


    return 0;
}
