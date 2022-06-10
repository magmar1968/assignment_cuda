#ifndef __PROCESS__
#define __PROCESS__

#include "../support_lib/myRandom/myRandom.cuh"

class Process
{

  public:
    //default constructor
    Process(){};

    Process(rnd::MyRandom * gnr)
        :_gnr(gnr)
    {}
    virtual ~Process(){}
    //functions
    double Get_random_uniform()
    {
        if(_gnr->Get_status())
            return _gnr->genUniform();
        else 
            exit(2);
    }

    double Get_random_gaussian()
    {
      if(_gnr -> Get_status())
          return _gnr->genGaussian();
      else 
          exit(2);
    }
  private:
    rnd::MyRandom * _gnr; //random number generator
};






#endif