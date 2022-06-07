#ifndef __TAUSWORTH__
#define __TAUSWORTH__

#include "../myRandom.cuh"

namespace rnd
{
class GenTausworth : public MyRandomImplementation
  {
    protected:
    public:
      HD GenTausworth(){};
      /**
       * @brief Tausoworth method for random numbers generation. Three different parametres
       *        set are available.
       * 
       * @param seed 
       * @param type TAUSWORTH_[1-2]
       */
      HD GenTausworth(const uint seed,const uint type = TAUSWORTH_1);
      HD uint  genUniformInt();
      HD ~GenTausworth(){};
    private: 
      uint _type,_k1,_k2,_k3,_m;
      uint _current;
      //tausworth parametres
      static const uint TAUS_1_K1 = 13U;
      static const uint TAUS_1_K2 = 19U;
      static const uint TAUS_1_K3 = 12U;
      static const uint TAUS_1_M  = 4294967294UL;

      static const uint TAUS_2_K1 = 2U;
      static const uint TAUS_2_K2 = 25U;
      static const uint TAUS_2_K3 = 4U;
      static const uint TAUS_2_M  = 4294967288UL;

      static const uint TAUS_3_K1 = 3U;
      static const uint TAUS_3_K2 = 11U;
      static const uint TAUS_3_K3 = 17U;
      static const uint TAUS_3_M  = 4294967280UL;
  };

}

#endif