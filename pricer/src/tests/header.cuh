/**
 * @file   header.cuh
 * @author Lorenzo Magnoni Andrea Ripamonti 
 * @brief  this file contain all the src file of the library
 * @version 0.1
 * @date 2022-09-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "../lib/support_lib/exact_eval_lib/exact_evaluation_formulas.cuh"
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
#include "../lib/contract_option_lib/contract_eq_option_esotic/contract_eq_option_esotic_corridor.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/myDouble_lib/myudouble.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/support_lib/timer_lib/myTimer.cuh"



#define N_TEST_SIM 1






