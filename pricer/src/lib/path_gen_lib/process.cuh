#ifndef __PROCESS__
#define __PROCESS__

#include "../support_lib/myRandom/myRandom.cuh"
#include "../equity_lib/equity_prices.cuh"
namespace prcr{

  // cuda macro
  #define H __host__
  #define D __device__
  #define HD __host__ __device__

  class Process
  {
    private:
      rnd::MyRandom * _gnr; //random number generator
    public:
      //default constructor
      HD Process(){};

      HD Process(rnd::MyRandom * gnr)
          :_gnr(gnr)
      {}
      HD virtual ~Process(){}
      //functions 

      HD double Get_random_gaussian()
      {
        if(_gnr -> Get_status())
          return _gnr->genGaussian();
        else 
          return -100;
      }

      HD double Get_random_uniform() //??
      {
        if(_gnr->Get_status())
          return _gnr->genUniform();
        else
          return -100;
          // exit(2);
      }

      HD virtual double Get_new_eq_price(Equity_description * eq_descr,
                        double eq_price,
                        double w,
                        double delta_t)=0;


  };
}






#endif
