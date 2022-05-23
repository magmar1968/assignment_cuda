#include "myRandom.hpp"
namespace rnd
{
    HOST_DEVICE MyRandomImplementation::MyRandomImplementation(uint m)
    :_m(m)
    {
    }

    HOST_DEVICE void MyRandomImplementation::setGaussImpl(const uint type)
    {
        _type = type;
    }

    HOST_DEVICE void MyRandomImplementation::setM(const uint m)
    {
        _m = m;
    }

    HOST_DEVICE double MyRandomImplementation::genUniform(const double min, const double max)
    {  
        return genUniformInt()/(double)_m * (max - min) + min;
    }

    HOST_DEVICE double MyRandomImplementation::genGaussian(const double mean, const double dev_std)
    {
        if(_storedValue)
        {
            _storedValue = false;
            return _value;
        }
        else
        {
            double num = 0,r = 0,u,v;
            switch (_type)
            {
            case GAUSSIAN_1:
                u = genUniform(); v = genUniform(); //input numbers
                num    = (sqrt(-2 * log( u) ) * cos( v * (2 * M_PI)));
                _value = (sqrt(-2 * log( u) ) * sin( v * (2 * M_PI)));
                break;
            
            case GAUSSIAN_2:
                while(r == 0 or r >= 1)
                {
                   u = genUniform(-1,1); v = genUniform(-1,1);
                   r = u*u + v*v; 
                }
                num    = u * sqrt(-2. * log(r) / r);
                _value = v * sqrt(-2. * log(r) / r);
                break;
            default:
                // std::cerr << "ERROR: in __function__\n"
                //       << "           wrong input available on Gaussian_(1-2)\n";
                break;
            }         
            _storedValue  = true;
            _value = _value*dev_std + mean;
            //normalize the number for the required mean and dev_std
            _value = _value*dev_std + mean;
            return  num * dev_std  + mean;
        }
    }

    //------------------------------------------------------------------------------------

    HOST_DEVICE GenLinCongruential::GenLinCongruential(uint seed, uint a, uint b, uint m )
        :_current(seed),_a(a),_b(b),_m(m),MyRandomImplementation(m)
    {
    }

    HOST_DEVICE uint GenLinCongruential::genUniformInt()
    {
        return _current = ( _a * _current + _b) % _m;
    }

    //--------------------------------------------------------------------------------------

    HOST_DEVICE GenTausworth::GenTausworth(uint seed, uint type)
        :_current(seed)
    {
        if(seed < 128)
        {
            // std::cerr<< "ERROR: in __FUNCTION__             \n"
            //            << "       seed must be grater than 128\n";
            _status = false;
        }

        // parameter settings

        switch (type)
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
            // std::cerr << "ERROR: wrong tausworth input please use one of\n"
            //           << "       the avaible macro TAUSWORTH_(0-2)      \n";
            break;
        }

        setM(_m);

    }

    HOST_DEVICE uint GenTausworth::genUniformInt()
    {
        uint b    = ((_current << _k1) ^ _current ) >> _k2;
        return _current  = ((_current & _m ) << _k3) ^ b;
    }

    HOST_DEVICE bool GenTausworth::getStatus() const
    {
        return _status;
    }

    // ---------------------------------------------------------------------------------------------

    HOST_DEVICE GenCombined::GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m)
        :_seed1(seed1), _seed2(seed2), _seed3(seed3), _seed4(seed4), _m(m),
        MyRandomImplementation(m)
    {
        genT1 = GenTausworth(_seed1, TAUSWORTH_1);
        genT2 = GenTausworth(_seed2, TAUSWORTH_2);
        genT3 = GenTausworth(_seed3, TAUSWORTH_3);

        genL1 = GenLinCongruential(_seed4);

        if(!genT1.getStatus() and genT2.getStatus() and genT3.getStatus())
        {
            // std::cerr << "ERROR: in __FUNCTION__";
            _status = false;
        }
    }


    HOST_DEVICE uint GenCombined::genUniformInt()
    {
        return genT1.genUniformInt()^genT2.genUniformInt()^
               genT3.genUniformInt()^genL1.genUniformInt();
    }


}