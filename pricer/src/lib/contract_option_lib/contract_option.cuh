#ifndef __CONTRACT_OPTION__
#define __CONTRACT_OPTION__

#include "../equity_lib/schedule_lib/schedule.cuh"


#define HD __host__ __device__ 

class Contract_option
{
  private:
    Schedule * _schedule ;

  public:
    //constructor & destructor 
    HD Contract_option(void){}
    HD Contract_option(Schedule *schedule)
        :_schedule(schedule)
        {}
    HD virtual ~Contract_option(void) {}

    HD Schedule * Get_schedule()const{return _schedule;}
};





#endif