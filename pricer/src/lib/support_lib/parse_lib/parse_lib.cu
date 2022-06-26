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
    H bool fileOptionExist(std::string fileName, 
                           std::string option, 
                           std::string *_line)
    {
        std::fstream ifs(fileName, std::fstream::in);

        bool found = false;
        std::string * line = new std::string;
        std::string appo;
        if(!ifs.is_open())
        {
            std::cerr << "ERROR: unable to open the file input\n"
                      << "       please check the file name   \n";
            exit(-1);
        }
        while(!ifs.bad() and !ifs.eof())
        {
            std::getline(ifs,*line);
            // eliminate line of comments and blank lines
            if(line->size() == 0)
                continue;
            
            if(line->find("!") != std::string::npos)
            {
                if(line->rfind("!")== 0)
                    continue;
                else{
                    line->resize(line->find("!"));
                }
            }

            if(!(line->rfind("#",0)==0))//basically a don't start with
                continue;

            //look for the option
            if(line->find(option) != std::string::npos)
            {
                found = true;
                *_line = *line;
                break;
            }
        }
        return found;
    }

}


