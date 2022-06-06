#ifndef __LIN_CONGRUENTIAL__
#define __LIN_CONGRUENTIAL__

#include "../myRandom.cuh"

namespace rnd
{
   
  class GenLinCongruential : public MyRandomImplementation
  {
    protected:
    public:
      static const uint DEFAULT_A = 1664525;
      static const uint DEFAULT_B = 1013904223;
      HD GenLinCongruential(){};
      /**
       * @brief Linnear Generator Congruential object that generate random
       *        number following the rule: S_{i+1} = (a* S_i + b) mod m
       * 
       * @param seed 
       * @param a (default 1664525)
       * @param b (default 1013904223)
       * @param m (default MAX_INT) 
       */
      HD GenLinCongruential(uint seed, uint a = DEFAULT_A, uint b = DEFAULT_B , uint m = UINT_MAX);
      HD ~GenLinCongruential(){};
      HD uint genUniformInt();
    private:
      uint _a, _b, _m; 
      uint _current;
  };
}

#endif