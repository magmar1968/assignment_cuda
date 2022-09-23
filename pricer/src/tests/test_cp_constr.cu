#include <iostream>
#include "device_launch_parameters.h"
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_term_structure.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/option_pricer_lib/option_pricer.cuh"
#include "../lib/option_pricer_lib/option_pricer_montecarlo/option_pricer_montecarlo.cuh"
#include "../lib/contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"



int main()
{
    using namespace prcr;

    Yield_curve_flat * yc_curve = new Yield_curve_flat("EUR",0.2);
    Volatility_surface * sigma = new Volatility_surface(0.01);
    Equity_description * eq_descr = new Equity_description("code","nome","EUR",0.3,
                                                            yc_curve,sigma);  
                                                            
    Equity_prices eq_price1(0,100,eq_descr);

    Equity_prices * eq_price2 = new Equity_prices;

    *eq_price2 = eq_price1;

    std::cout << eq_price2->Get_eq_price().get_number() << std::endl;
    return 0;
}





















