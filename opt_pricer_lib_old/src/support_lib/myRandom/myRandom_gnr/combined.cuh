#ifndef __COMBINED__
#define __COMBINED__

#include "../myRandom.cuh"
#include "tausworth.cuh"
#include "linCongruential.cuh"

namespace rnd
{
    class GenCombined : public MyRandomImplementation
  {
    public:
      HD GenCombined();
      /**
       * @brief combined three tausworth generator and one linear congruential to get a longer period. Needs a seed
       *        for each of the four generators.
       * @param seed1  (tausworth)
       * @param seed2  (tausworth)
       * @param seed3  (tausworth)
       * @param seed4  (lin congruential)
       * @param m 
       */
      HD GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m = UINT_MAX);
      HD ~GenCombined(){};
      HD uint genUniformInt();

    private:
      uint _seed1, _seed2, _seed3, _seed4, _m;
      uint _current;

      GenTausworth genT1, genT2, genT3;
      GenLinCongruential genL1;

      HD void genSeeds();
      HD void genSeeds(const uint seed);
  };
}


#endif