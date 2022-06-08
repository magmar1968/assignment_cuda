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
        HD myudouble(double number);

        HD double get_number() const;
        //HD void set_number(double number); //utile?
        HD bool check_sign() const;

        //operators overloading
        myudouble& operator+=(const myudouble& rhs);
        
        myudouble& operator-=(const myudouble& rhs);

        double* operator*(const myudouble& rhs)
        {
            return &_number;
        }


    private:
        double  _number;
       
    };
    typedef pricer::myudouble udb;

    inline myudouble operator+(myudouble& lhs, const myudouble& rhs)
    {
        lhs += rhs;
        return lhs;
    }
    inline myudouble operator+(myudouble& lhs, const myudouble& rhs)
    {
        lhs += myudouble(rhs);
        return lhs;
    }

    inline bool operator==(const myudouble& lhs, const myudouble& rhs)
    { 
        return lhs.get_number() == rhs.get_number() ? true : false;
    }
    inline bool operator!=(const myudouble& lhs, const myudouble& rhs){return !operator==(lhs,rhs);}
    inline bool operator< (const myudouble& lhs, const myudouble& rhs)
    {
        return lhs.get_number() < rhs.get_number() ? true : false;     
    }
    inline bool operator> (const myudouble& lhs, const myudouble& rhs){return  operator< (rhs,lhs);}
    inline bool operator<=(const myudouble& lhs, const myudouble& rhs){return !operator> (lhs,rhs);}
    inline bool operator>=(const myudouble& lhs, const myudouble& rhs){return !operator< (lhs,rhs);}



}


#endif