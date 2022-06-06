#include "linCongruential.cuh"

namespace rnd
{
    HD GenLinCongruential::GenLinCongruential(uint seed, uint a, uint b, uint m )
        :_current(seed),_a(a),_b(b),_m(m),MyRandomImplementation(m)
    {
    }

    HD uint GenLinCongruential::genUniformInt()
    {
        return _current = ( _a * _current + _b) % _m;
    }
}