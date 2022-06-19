#ifndef __MYUDOUBLE__
#define __MYUDOUBLE__

/*Classe per i prices, option prices must be >0*/
namespace pricer
{


#define H __host__
#define D __device__
#define HD __host__ __device__   



    class myudouble
    {
    public:
        HD myudouble() {};
        HD myudouble(double number);

        HD double get_number() const;
        //HD void set_number(double number); //utile?
        HD bool check_sign() const;

        //operators overloading

        HD void operator=(const myudouble& rhs);
        HD void operator=(const double& rhs);

        HD myudouble& operator+=(const myudouble& rhs);
        HD myudouble& operator-=(const myudouble& rhs);
        HD myudouble& operator*=(const myudouble& rhs);

        // HD double& operator*(const myudouble& rhs)
        // {
        //     return _number;
        // }


    private:
        double  _number;      
    };
    typedef pricer::myudouble udb;

    HD inline myudouble operator+(myudouble& lhs, const myudouble& rhs)
    {
        lhs += rhs;
        return lhs;
    }
    HD inline myudouble operator+(myudouble& lhs, const double& rhs)
    {
        lhs += myudouble(rhs);
        return lhs;
    }
    HD inline myudouble operator+(const double& lhs, myudouble& rhs)
    {
        rhs += myudouble(lhs);
        return rhs;
    }
    HD inline myudouble operator*(myudouble& lhs, const double& rhs)
    {
        lhs *= myudouble(rhs);
        return lhs;
    }
    HD inline myudouble operator*(const double& lhs, myudouble& rhs)
    {
        rhs *= myudouble(lhs);
        return rhs;
    }
    HD inline bool operator==(const myudouble& lhs, const myudouble& rhs)
    { 
        return lhs.get_number() == rhs.get_number() ? true : false;
    }
    HD inline bool operator!=(const myudouble& lhs, const myudouble& rhs){return !operator==(lhs,rhs);}
    HD inline bool operator< (const myudouble& lhs, const myudouble& rhs)
    {
        return lhs.get_number() < rhs.get_number() ? true : false;     
    }
    HD inline bool operator> (const myudouble& lhs, const myudouble& rhs){return  operator< (rhs,lhs);}
    HD inline bool operator<=(const myudouble& lhs, const myudouble& rhs){return !operator> (lhs,rhs);}
    HD inline bool operator>=(const myudouble& lhs, const myudouble& rhs){return !operator< (lhs,rhs);}

}


#endif