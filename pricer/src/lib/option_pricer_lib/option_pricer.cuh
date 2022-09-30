#ifndef __OPTION_PRICER__
#define __OPTION_PRICER__

#include "../contract_option_lib/contract_option.cuh"
#include "../path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"

namespace prcr
{
  #define HD __host__ __device__

  class Option_pricer
  {
    protected:// è una buona pratica?
      Contract_option      * _contract_option ;
      Process_eq_lognormal * _process ;

    public:
  // Constructors & destructors
      HD Option_pricer(void){}
      HD Option_pricer(Contract_option      *contract_option,
                      Process_eq_lognormal  *process)
          :_contract_option(contract_option),_process(process)
      {}
      HD virtual ~Option_pricer(void) {};
  } ;

}


#endif