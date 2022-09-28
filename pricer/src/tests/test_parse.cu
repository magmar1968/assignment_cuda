#include "../lib/support_lib/parse_lib/parse_lib.cuh"


int main()
{
    using namespace prcr;
    Pricer_args * prcr_args = new Pricer_args;

    bool status = ReadInputOption("./data/input_file_template.txt",prcr_args);
    

    if (status == true)
        std::cout << "No errors encountered" << std::endl;
    if (status == false)
        std::cout << "An error was encountered" << std::endl;
        

}