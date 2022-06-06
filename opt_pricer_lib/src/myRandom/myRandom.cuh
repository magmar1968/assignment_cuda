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
#define _USE_MATH_DEFINES

namespace rnd
{
    typedef unsigned int uint; //windows
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

    typedef unsigned int uint; //for windows
    
    //############################################################################################ 
    
    //      abstract class 
    class MyRandom
    {
      public:
        HD MyRandom(){};
        HD ~MyRandom(){};

        HD virtual double genUniform(const double min = 0, const double max = 1) = 0;
        // HD virtual double* genUniformVector(
        //                           const size_t dim,
        //                           const double min = 0.,
        //                           const double max = 1.)=0;
        HD virtual double genGaussian(const double mean = 0, const double dev_std = 1) = 0;
        // HD virtual double* genGaussianVector(
        //                           const size_t dim,
        //                           const double mean = 0.,
        //                           const double dev_std = 1.)=0;
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
      /**
       * @brief generate gaussian numbers from a uniform distribution. Use both the possible 
       *        box-muller formulas. Default is the trigonometric one.
       * @param mean 
       * @param dev_std 
       * @return double 
       */
      HD double genGaussian(const double mean = 0., const double dev_std = 1.);
      // HD double* genGaussianVector(
      //                           const size_t dim,
      //                           const double mean = 0.,
      //                           const double dev_std = 1.);

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
  
  
  //seeds generetor functions
  H uint genSeed(bool tausworth = false); // tausworth seed must be > 128
}


#endif