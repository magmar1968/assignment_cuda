#include "./parse_lib.cuh"

namespace prcr
{

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

    //file options 


    H bool fileOptionExist(std::string fileName, std::string option)
    {
        std::fstream ifs(fileName, std::fstream::in);

        bool found = false;
        std::string line;
        while(!ifs.bad() and !ifs.eof())
        {
            std::getline(ifs,line,'!');
            if(!(line.rfind("#",0)==0))//basically don't start with
                continue;
            found = line.find(option);
            if(found) break;
        }

        return found;
    }











}


