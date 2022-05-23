#ifndef __MYRANDOM__
#define __MYRANDOM__


#include <iostream> // cout
#include <math.h>   // sin, cos
#include <climits>  //INT_MAX




namespace rnd
{
    #define HOST __host__
    #define DEVICE __device__
    #define HOST_DEVICE __host__ __device__ 





    #define GAUSSIAN_1 0
    #define GAUSSIAN_2 1

    #define TAUSWORTH_1 0
    #define TAUSWORTH_2 1
    #define TAUSWORTH_3 2

    #define TAUS_1_K1 13
    #define TAUS_1_K2 19
    #define TAUS_1_K3 12
    #define TAUS_1_M  4294967294UL

    #define TAUS_2_K1 2
    #define TAUS_2_K2 25
    #define TAUS_2_K3 4
    #define TAUS_2_M  4294967288UL

    #define TAUS_3_K1 3
    #define TAUS_3_K2 11
    #define TAUS_3_K3 17
    #define TAUS_3_M  4294967280UL
    // typedef unsigned int uint; for windows
    
    //---------------------------------------------------------------------------------------- 
    
    //      abstract class 
    class MyRandom
    {
      public:
        HOST_DEVICE MyRandom(){};
        HOST_DEVICE ~MyRandom(){};
      
        HOST_DEVICE virtual double genUniform(const double min = 0, const double max = 1) = 0;
        
        HOST_DEVICE virtual double genGaussian(const double mean = 0, const double dev_std = 1) = 0;
      protected:
        HOST_DEVICE virtual uint genUniformInt() = 0;
    };


  class MyRandomImplementation : public MyRandom
  {
    public:
      HOST_DEVICE double genUniform(const double min = 0., const double max = 1.);
      HOST_DEVICE double genGaussian(const double mean = 0., const double dev_std = 1.);
      HOST_DEVICE void   setGaussImpl(const uint type);
  
    protected: // accessible by all subclasses
      HOST_DEVICE MyRandomImplementation(uint m=UINT_MAX);
      HOST_DEVICE void setM(uint m);
    private:
      bool    _storedValue = false;
      double  _value;
      uint    _m, _type = GAUSSIAN_1;
  };
      

  class GenLinCongruential : public MyRandomImplementation
  {
    protected:
    public:
      static const uint DEFAULT_A = 1664525;
      static const uint DEFAULT_B = 1013904223;
      HOST_DEVICE GenLinCongruential(){};
      /**
       * @brief Construct a new Linnear Generator Congruential object that generate random
       *        number following the rule: S_{i+1} = (a* S_i + b) mod m
       * 
       * @param seed 
       * @param a (default 1664525)
       * @param b (default 1013904223)
       * @param m (default MAX_INT) 
       */
      HOST_DEVICE GenLinCongruential(uint seed, uint a = DEFAULT_A, uint b = DEFAULT_B , uint m = UINT_MAX);
      HOST_DEVICE ~GenLinCongruential(){};
    private:
      uint _a, _b, _m; 
      uint _current;
      friend class GenCombined; //to get access to genUnifomInt
      HOST_DEVICE uint genUniformInt();
  };

  

  class GenTausworth : public MyRandomImplementation
  {
    protected:
    public:
      HOST_DEVICE GenTausworth(){};
      HOST_DEVICE GenTausworth(uint seed, uint type);
      HOST_DEVICE ~GenTausworth(){};
      HOST_DEVICE bool getStatus() const;
    private: 
      uint _k1,_k2,_k3,_m;
      uint _current;
      bool _status = true;
      friend class GenCombined; // to get access to genUnifomInt
      HOST_DEVICE uint  genUniformInt();
  };

  class GenCombined : public MyRandomImplementation
  {
    public:
      HOST_DEVICE GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m = UINT_MAX);
      HOST_DEVICE ~GenCombined(){};
      HOST_DEVICE bool getStatus() const;

    private:
      uint _seed1, _seed2, _seed3, _seed4, _m;
      uint _current;

      GenTausworth genT1, genT2, genT3;
      GenLinCongruential genL1;
      bool _status = true;
      HOST_DEVICE uint genUniformInt();
  };
}


#endif