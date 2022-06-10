#ifndef __RANDOM_NUMBERS__
#define __RANDOM_NUMBERS__

class Random_numbers
{
  private:
    double * _rnd_num;
    size_t   _dim;
  public:
    Random_numbers(){}
    Random_numbers(size_t dim)
        :_dim(dim)
    {
    }

    void Set_element(size_t i,double num)
    {
        if(i < _dim)
            _rnd_num[i] = num;
        else
            exit(1);        
    }

    double Get_element(size_t i) const
    {
        return _rnd_num[i];
    }
    double Get_element() const
    {
        return Get_element(0);
    }
};

#endif