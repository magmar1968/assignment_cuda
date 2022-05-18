#include "myRandom.h"
// #define DEBUG
namespace rnd
{
    double MyRandomImplementation::genUniform(const double min, const double max)
    {  
        uint n = genUniformInt();
        #ifdef DEBUG
        double m = n/(double)_m;
        double p = m* (max - min);
        std::cerr << "max - min      "  << (max - min) << std::endl;
        std::cerr << "rand int       "  << n << std::endl;
        std::cerr << "n/UINT_MAX     "  << m << std::endl;
        std::cerr << "UINT_MAX and m "  << UINT_MAX << "  " << _m << std::endl;
        std::cerr << "m* (max - min) "  << p << std::endl;
        std::cerr << "p + min        "  << p + min << std::endl;
        std::cerr << "check          "  << n/(double)_m * (max - min) + min << "\n\n";
        #endif
        
        return n/(double)_m * (max - min) + min;
    }

    double MyRandomImplementation::genGaussian(const double mean, const double dev_std)
    {
        if(_storedValue)
        {
            _storedValue = false;
            return _value;
        }
        else
        {
            //insert second way 
            double u = genUniform(), v = genUniform();
            double num = (sqrt(-2 * log( u) ) * cos( v * (2 * M_PI)));
            _value =  (sqrt(-2 * log( u) ) * sin( v * (2 * M_PI))); //check
            _value = _value*dev_std + mean;
            _storedValue  = true;
            //normalize the number for the required mean and dev_std 
            return  num * dev_std  + mean;   
        }
    }


    GenLinCongruential::GenLinCongruential(uint seed, uint a, uint b, uint m )
        :_current(seed),_a(a),_b(b),_m(m)
    {
        MyRandomImplementation::setM(_m);
    }

    uint GenLinCongruential::genUniformInt()
    {
        return _current = ( _a * _current + _b) % _m;
    }

    GenTausworth::GenTausworth(uint seed, uint type, uint m)
        :_current(seed), _m(m)
    {
        MyRandomImplementation::setM(_m);
        if(seed < 128)
        {
            std::cerr<< "ERROR: in __FUNCTION__             \n"
                       << "       seed must be grater than 128\n";
            _status = false;
        }

        // parameter settings

        switch (type)
        {
        case TAUSWORTH_1:
            _k1 = TAUS_1_K1;
            _k2 = TAUS_1_K2;
            _k3 = TAUS_1_K3;
            break;
        case TAUSWORTH_2:
            _k1 = TAUS_2_K1;
            _k2 = TAUS_2_K2;
            _k3 = TAUS_2_K3;
            break;
        case TAUSWORTH_3:
            _k1 = TAUS_3_K1;
            _k2 = TAUS_3_K2;
            _k3 = TAUS_3_K3;       
        default:
            std::cerr << "ERROR: wrong tausworth input please use one of\n"
                      << "       the avaible macro TAUSWORTH_(0-2)      \n";
            break;
        }

    }

    uint GenTausworth::genUniformInt()
    {
        uint b    = (((_current << _k1) ^ _current ) >> _k2);
        return _current  = (((_current & _m ) << _k3) ^ b);
    }

    bool GenTausworth::getStatus() const
    {
        return _status;
    }

    // ---------------------------------------------------------------------------------------------

    GenCombined::GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m)
        :_seed1(seed1), _seed2(seed2), _seed3(seed3), _seed4(seed4), _m(m)
    {
        MyRandomImplementation::setM(_m);
        genT1 = GenTausworth(_seed1, TAUSWORTH_1, _m);
        genT2 = GenTausworth(_seed2, TAUSWORTH_2, _m);
        genT3 = GenTausworth(_seed3, TAUSWORTH_3, _m);

        genL1 = GenLinCongruential(_seed4);

        if(!genT1.getStatus() and genT2.getStatus() and genT3.getStatus())
        {
            std::cerr << "ERROR: in __FUNCTION__";
            _status = false; 
        }
    }


    uint GenCombined::genUniformInt()
    {
        return genT1.genUniformInt()^genT2.genUniformInt()^
               genT3.genUniformInt()^genL1.genUniformInt();
    }







}