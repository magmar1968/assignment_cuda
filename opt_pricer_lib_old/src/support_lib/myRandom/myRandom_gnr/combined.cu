#include "combined.cuh"

namespace rnd
{
    HD GenCombined::GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m)
        :_seed1(seed1), _seed2(seed2), _seed3(seed3), _seed4(seed4), _m(m),
        MyRandomImplementation(m)
    {
        genT1 = GenTausworth(_seed1, TAUSWORTH_1);
        genT2 = GenTausworth(_seed2, TAUSWORTH_2);
        genT3 = GenTausworth(_seed3, TAUSWORTH_3);

        genL1 = GenLinCongruential(_seed4);

        if(!genT1.getStatus() && genT2.getStatus() && genT3.getStatus())
        {
            // std::cerr << "ERROR: in __FUNCTION__";
            _status = false;
        }
    }

    HD uint GenCombined::genUniformInt()
    {
        return genT1.genUniformInt()^genT2.genUniformInt()^
               genT3.genUniformInt()^genL1.genUniformInt();
    }

}