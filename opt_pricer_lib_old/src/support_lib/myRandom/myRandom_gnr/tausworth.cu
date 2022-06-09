#include "tausworth.cuh"

namespace rnd
{
        HD GenTausworth::GenTausworth(const uint seed, const uint type)
        :_current(seed),_type(type)
    {
        if(seed < 128)
        {
            _status = false;
        }

        // parameter settings
        switch (_type)
        {
        case TAUSWORTH_1:
            _k1 = TAUS_1_K1;
            _k2 = TAUS_1_K2;
            _k3 = TAUS_1_K3;
            _m  = TAUS_1_M;
            break;
        case TAUSWORTH_2:
            _k1 = TAUS_2_K1;
            _k2 = TAUS_2_K2;
            _k3 = TAUS_2_K3;
            _m  = TAUS_2_M;
            break;
        case TAUSWORTH_3:
            _k1 = TAUS_3_K1;
            _k2 = TAUS_3_K2;
            _k3 = TAUS_3_K3;
            _m  = TAUS_3_M;
            break;
        default:
            _status = false;
            break;
        }
        setM(_m);
    }

    HD uint GenTausworth::genUniformInt()
    {
        uint b    = ((_current << _k1) ^ _current ) >> _k2;
        return _current  = ((_current & _m ) << _k3) ^ b;
    }

}