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
            delete(line);
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
        delete(line);
        return found;
    }




    __host__ bool
    ReadInputOption(std::string filename, 
                    Pricer_args * prcr_args)
    {

        if(file_exists(filename) == false){
            std::cerr << "INPUT ERROR: the input file doesn't exist. Please check the\n"
                      << "              filname and retry.                           \n";
        }


        bool status = true;
        
        
        //dev options input
        status = status && fileGetOptionValue<bool>(filename,"#CPU",&prcr_args->dev_opts.CPU);
        status = status && fileGetOptionValue<bool>(filename,"#GPU",&prcr_args->dev_opts.GPU);
        if(prcr_args->dev_opts.CPU == false and prcr_args->dev_opts.GPU == false)
        {
            std::cerr << "INPUT ERROR: at least one between CPU and GPU must be set to true.\n"
                      << "             Please check your input file and retry.              \n";
            status =  false;
        }


        status = status && fileGetOptionValue<size_t>(filename, "#N_blocks",  &prcr_args->dev_opts.N_blocks);
        status = status && fileGetOptionValue<size_t>(filename, "#N_threads", &prcr_args->dev_opts.N_threads);

        //-----------------------------------------------------------------------------------------------------
        //MC options 
        status = status && fileGetOptionValue<size_t>(filename,"#N_simulations",&prcr_args->mc_args.N_simulations);


        //schedule options 
        double T;
        size_t steps;

        status = status && fileGetOptionValue<double>(filename, "#T",&T);
        status = status && fileGetOptionValue<size_t>(filename, "#m",&steps);

        prcr_args->schedule_args.T = T;        
        prcr_args->schedule_args.dim = steps; //devi considerare anche lo 0
        prcr_args->schedule_args.t_ref = 0.; //non ha senso fare altrimenti
        prcr_args->schedule_args.deltat = T/ static_cast<double>(steps);

        
        //-----------------------------------------------------------------------------------------------------
        //equity description arguments
        status = status && fileGetOptionValue<double>(filename,"#eq_descr_dy",&prcr_args->eq_descr_args.dividend_yield);
        status = status && fileGetOptionValue<double>(filename,"#volatility", &prcr_args->eq_descr_args.vol);
        status = status && fileGetOptionValue<double>(filename,"#yc_rate",    &prcr_args->eq_descr_args.rate);
        
        
        
        //-----------------------------------------------------------------------------------------------------
        //equity price arguments 
        status = status && fileGetOptionValue<double>(filename, "#eq_price_time",  &prcr_args->eq_price_args.time);
        status = status && fileGetOptionValue<double>(filename, "#eq_price_price", &prcr_args->eq_price_args.price);

        //-----------------------------------------------------------------------------------------------------
        //contract options 
        std::string option_type;
        status = status && fileGetOptionValue<std::string>(filename, "#option_type", &option_type);
        status = status && fileGetOptionValue<char>(filename, "#contract_type", &prcr_args->contract_args.contract_type);
        strcpy(prcr_args->contract_args.option_type, option_type.c_str());
        
        if(option_type == "vanilla"){
            status = status && fileGetOptionValue<double>(filename, "#strike_price", &prcr_args->contract_args.strike_price);
        }
        else if(option_type == "esotic_corridor"){
            status = status && fileGetOptionValue<double>(filename, "#strike_price", &prcr_args->contract_args.strike_price);
            status = status && fileGetOptionValue<double>(filename, "#B", &prcr_args->contract_args.B );
            status = status && fileGetOptionValue<double>(filename, "#Noz", &prcr_args->contract_args.N );
            status = status && fileGetOptionValue<double>(filename, "#K", &prcr_args->contract_args.K );    
        }
        else{
            std::cerr << "INPUT ERROR: the contract option doesn't exist. Please check the input\n"
                      << "             file and retry.                                          \n";
        }

        //-----------------------------------------------------------------------------------------------------
        // stocastic process options 
        status = status && fileGetOptionValue<bool>(filename, "#pr_exact",&prcr_args->stc_pr_args.exact);
        status = status && fileGetOptionValue<size_t>(filename,"#l",&prcr_args->stc_pr_args.l);
                
        return status;
    }
}


