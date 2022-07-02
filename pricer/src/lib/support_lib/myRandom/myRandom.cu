#include "myRandom.cuh"


namespace rnd
{
    HD bool MyRandom::Get_status() const
    {
        return _status;
    }

    HD void MyRandom::Set_status(bool status) 
    {
        _status = status;
    }


    HD MyRandomImplementation::MyRandomImplementation(uint m)
    :_m(m)
    {
        _storedValue = false;
        _type = GAUSSIAN_1;
    }

    HD void MyRandomImplementation::setGaussImpl(const uint type)
    {
        _type = type;
    }

    HD void MyRandomImplementation::setM(const uint m)
    {
        _m = m;
    }

    HD double MyRandomImplementation::genUniform(const double min, const double max)
    {  
        return genUniformInt()/(double)_m * (max - min) + min;
    }

    HD double MyRandomImplementation::genGaussian(const double mean, const double dev_std)
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
                while (u == 0)
                {
                    u = genUniform();
                }
                num    = (sqrt(-2 * log( u) ) * cos( v * (2 * M_PI)));
                _value = (sqrt(-2 * log( u) ) * sin( v * (2 * M_PI)));
                break;
            
            case GAUSSIAN_2:
                while(r == 0 || r >= 1)
                {
                   u = genUniform(-1,1); v = genUniform(-1,1);
                   r = u*u + v*v; 
                }
                num    = u * sqrt(-2. * log(r) / r);
                _value = v * sqrt(-2. * log(r) / r);
                break;
            default:
                Set_status(false);
                break;
            }         
            _storedValue  = true;
            //normalize the number for the required mean and dev_std
            _value = _value*dev_std + mean;                
            return  num * dev_std  + mean;
        }
    }


    //---------------------------------------------------------------------------------------
    H uint genSeed(bool tausworth )
    {
        uint seed = rand();
        while(seed < 128 && tausworth)
        {
            seed = rand();
        }
        return seed;
    }
}