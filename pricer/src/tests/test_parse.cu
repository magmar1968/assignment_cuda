#include "../lib/support_lib/parse_lib/parse_lib.cuh"


int main()
{
    using namespace prcr;
    Pricer_args * prcr_args = new Pricer_args;

    bool status = ReadInputOption("./data/input_file_template.txt",prcr_args);
    if( status == false){
        delete(prcr_args);
        return -1;
    }
    else{
        delete(prcr_args);
        return 0;
    }
        

}