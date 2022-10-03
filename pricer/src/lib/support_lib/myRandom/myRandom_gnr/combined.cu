#include "combined.cuh"

namespace rnd
{
    HD GenCombined::GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m)
        :_seed1(seed1), _seed2(seed2), _seed3(seed3), _seed4(seed4), _m(m),
        MyRandomImplementation(m)
    {
        GenTausworth genT1(_seed1, TAUSWORTH_1);
        GenTausworth genT2(_seed2, TAUSWORTH_2);
        GenTausworth genT3(_seed3, TAUSWORTH_3);

        GenLinCongruential genL1(_seed4);

        if(!genT1.Get_status() && genT2.Get_status() && genT3.Get_status())
        {
            // std::cerr << "ERROR: in __FUNCTION__";
            Set_status(false);
        }
    }

    HD uint GenCombined::genUniformInt()
    {
        return genT1.genUniformInt()^genT2.genUniformInt()^
               genT3.genUniformInt()^genL1.genUniformInt();
    }

}