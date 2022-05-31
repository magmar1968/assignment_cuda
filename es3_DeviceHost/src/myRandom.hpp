/**
 * @file myRandom.hpp
 * @author Lorenzo Magnoni/Andrea Ripamonti/Matteo Martelli (you@domain.com)
 * @brief  This class implement three different methods for the generation of random numbers. The class is
 *         structured to be usable also by the GPU. 
 *    
 * @version 1.0
 * @date 2022-05-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __MYRANDOM__
#define __MYRANDOM__


#include <iostream> // cout, endl
#include <math.h>   // sin, cos
#include <climits>  //INT_MAX
#include <vector>



namespace rnd
{
    // cuda macro
    #define H __host__
    #define D __device__
    #define HD __host__ __device__ 
    //gaussian method macros type
    #define GAUSSIAN_1 1
    #define GAUSSIAN_2 2
    //tausworth method macros type
    #define TAUSWORTH_1 1
    #define TAUSWORTH_2 2
    #define TAUSWORTH_3 3

    using std::vector;


    typedef unsigned int uint; //for windows
    
    //############################################################################################ 
    
    //      abstract class 
    class MyRandom
    {
      public:
        HD MyRandom(){};
        HD ~MyRandom(){};

        HD virtual double genUniform(const double min = 0, const double max = 1) = 0;
        HD virtual double genGaussian(const double mean = 0, const double dev_std = 1) = 0;

      protected:
        HD virtual uint genUniformInt() = 0;
    };

  /**
   * @brief implement the genGaussian and gen Uniform method. Also introduce the status class flag to
   *        check if the generator is usable. Constructor is not accesible by the user.
   */
  class MyRandomImplementation : public MyRandom
  {
    public:
      /**
       * @brief generate numbers according to the uniform destribution. 
       * @param min 
       * @param max 
       * @return double 
       */
      HD double genUniform(const double min = 0., const double max = 1.);
      HD double* genUniformVector(
                                   size_t dim,
                                  const double min = 0.,
                                  const double max = 1.);
      /**
       * @brief generate gaussian numbers from a uniform distribution. Use both the possible 
       *        box-muller formulas. Default is the trigonometric one.
       * @param mean 
       * @param dev_std 
       * @return double 
       */
      HD double genGaussian(const double mean = 0., const double dev_std = 1.);
      HD double* genGaussianVector(
                                const size_t dim,
                                const double mean = 0.,
                                const double dev_std = 1.);

      HD void   setGaussImpl(const uint type);
      HD bool   getStatus() const;
  
    protected: // accessible by all subclasses
      HD MyRandomImplementation(uint m=UINT_MAX); 
      HD void setM(uint m);
      bool    _status;
    
    private:
      bool    _storedValue;
      double  _value;
      uint    _m, _type;
  };
      
  class GenLinCongruential : public MyRandomImplementation
  {
    protected:
    public:
      static const uint DEFAULT_A = 1664525;
      static const uint DEFAULT_B = 1013904223;
      HD GenLinCongruential(){};
      /**
       * @brief Linnear Generator Congruential object that generate random
       *        number following the rule: S_{i+1} = (a* S_i + b) mod m
       * 
       * @param seed 
       * @param a (default 1664525)
       * @param b (default 1013904223)
       * @param m (default MAX_INT) 
       */
      HD GenLinCongruential(uint seed, uint a = DEFAULT_A, uint b = DEFAULT_B , uint m = UINT_MAX);
      HD ~GenLinCongruential(){};
      HD uint genUniformInt();
    private:
      uint _a, _b, _m; 
      uint _current;
  };

  

  class GenTausworth : public MyRandomImplementation
  {
    protected:
    public:
      HD GenTausworth(){};
      /**
       * @brief Tausoworth method for random numbers generation. Three different parametres
       *        set are available.
       * 
       * @param seed 
       * @param type TAUSWORTH_[1-2]
       */
      HD GenTausworth(const uint seed,const uint type = TAUSWORTH_1);
      HD uint  genUniformInt();
      HD ~GenTausworth(){};
    private: 
      uint _type,_k1,_k2,_k3,_m;
      uint _current;
      //tausworth parametres
      static const uint TAUS_1_K1 = 13U;
      static const uint TAUS_1_K2 = 19U;
      static const uint TAUS_1_K3 = 12U;
      static const uint TAUS_1_M  = 4294967294UL;

      static const uint TAUS_2_K1 = 2U;
      static const uint TAUS_2_K2 = 25U;
      static const uint TAUS_2_K3 = 4U;
      static const uint TAUS_2_M  = 4294967288UL;

      static const uint TAUS_3_K1 = 3U;
      static const uint TAUS_3_K2 = 11U;
      static const uint TAUS_3_K3 = 17U;
      static const uint TAUS_3_M  = 4294967280UL;
  };

  class GenCombined : public MyRandomImplementation
  {
    public:
      HD GenCombined();
      /**
       * @brief combined three tausworth generator and one linear congruential to get a longer period. Needs a seed
       *        for each of the four generators.
       * @param seed1  (tausworth)
       * @param seed2  (tausworth)
       * @param seed3  (tausworth)
       * @param seed4  (lin congruential)
       * @param m 
       */
      HD GenCombined(uint seed1, uint seed2, uint seed3, uint seed4, uint m = UINT_MAX);
      HD ~GenCombined(){};
      HD uint genUniformInt();

    private:
      uint _seed1, _seed2, _seed3, _seed4, _m;
      uint _current;

      GenTausworth genT1, genT2, genT3;
      GenLinCongruential genL1;

      HD void genSeeds();
      HD void genSeeds(const uint seed);
  };

  //seeds generetor functions
  H uint genSeed(bool tausworth = false); // tausworth seed must be > 128
}


#endif