#include <iostream> // cout
#include <math.h>   // sin, cos
#include <climits>  //INT_MAX


namespace rnd
{
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
            T num = (sqrt(-2 * log( u) ) * cos( v * (2 * M_1_PIf64)));
            //normalize the number for the required mean and dev_std 
            return  num * dev_std  + mean;   
        }
    };

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
        LinGenCongruential(int seed, uint a = 1664525, uint b = 1013904223, uint m = INT32_MAX)
            :_current(seed),_a(a),_b(b),_m(m)
            {};
        ~LinGenCongruential(){};

        T genUniform(const T min = 0, const T max = 1) 
        {
            _current = ( _a * _current + _b) % _m;
            T num = _current / (T) _m;
            return num * max + min;
        };
        

      private:
        uint _a, _b, _m; 
        uint _current;
    };








}
