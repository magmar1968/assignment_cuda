#ifndef __RANDOM_NUMBERS__
#define __RANDOM_NUMBERS__

#define HD __host__ __device__

class Random_numbers
{
  private:
    double * _rnd_num;
    size_t   _dim;
  public:
    HD Random_numbers(){}
    HD Random_numbers(size_t dim)
        :_dim(dim)
    {
        _rnd_num = new double[_dim];
    }
    HD ~Random_numbers()
    {
	delete[](_rnd_num);
    }
    HD void Set_element(size_t i,double num)
    {
        if (i < _dim)
            _rnd_num[i] = num;
        /*else
             exit(1);*/        
    }
    HD double Get_element(size_t i) const
    {
        return _rnd_num[i];
    }
    HD double Get_element() const
    {
        return Get_element(0);
    }
};

#endif
