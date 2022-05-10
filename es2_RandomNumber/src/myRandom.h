#include <iostream> // cout
#include <math.h>   // sin, cos
#include <climits>  //INT_MAX


namespace rnd
{


  // =========================
  //      abstract class 
    template<typename T>
    class MyRandom
    {
      public:
        MyRandom(){};
        ~MyRandom(){};
      
        virtual T genUniform(const T min = 0, const T max = 1) = 0;
        
        T genGaussian(const T mean = 0, const T dev_std = 1) const
        {
            T u = genUniform(), v = genUniform();
            T num = (sqrt(-2 * log( u) ) * cos( v * (2 * M_PI)));
            //normalize the number for the required mean and dev_std 
            return  num * dev_std  + mean;   
        }
    };

  // =========================
  //   linear congruential

  template< typename T>
  class LinGenCongruential : 
    public MyRandom<T>
  {
    public:
      /**
       * @brief Construct a new Linnear Generator Congruential object that generate random
       *        number following the rule: S_{i+1} = (a* S_i + b) mod m
       * 
       * @param seed 
       * @param a (default 1664525)
       * @param b (default 1013904223)
       * @param m (default MAX_INT) 
       */
      LinGenCongruential(int seed, uint a = 1664525, uint b = 1013904223, uint m = UINT_MAX)
        :_current(seed),_a(a),_b(b),_m(m)
      {};
      ~LinGenCongruential(){};
      T genUniform(const T min = 0, const T max = 1)
      {
          _current = ( _a * _current + _b) % _m;
          T num = _current / (T) _m;
          return num *(max - min) + min;
      }
    private:
      uint _a, _b, _m; 
      uint _current;
  };

  template<typename T>
  class GenTausworth :
    public MyRandom<T>
  {
    public:
      GenTausworth(uint seed = 256, uint k1 = 13, uint k2 = 19,uint k3 = 12, uint m = UINT_MAX)
        :_current(seed),_k1(k1),_k2(k2),_k3(k3),_m(m)
      {
          if(seed < 128)
          {
              _current += 128;
              std::cerr<< "ERROR: in __FUNCTION__             \n"
                       << "       seed must be grater than 128\n";
          } 
      }

      T genUniform(const T min = 0, const T max = 1)
      {
          uint b    = (((_current << _k1) ^ _current ) >> _k2);
          _current  = (((_current & _m ) << _k3) ^ b);
          T num = _current / (T)_m;
          return num * (max - min) + min;
        
      }

    private: 
      uint _k1,_k2,_k3,_m;
      uint _current;
  };
}
