#ifndef __CONTRACT_EQ_OPTION_ESOTIC__
#define __CONTRACT_EQ_OPTION_ESOTIC__

#include "../contract_eq_option.cuh"
#define HD __host__ __device__


class Contract_eq_option_esotic : public Contract_eq_option {
  private:
    double _strike_price ;
protected:
    char   _contract_type ; // C for call P for put
  public:
    // constructors & destructors
    HD Contract_eq_option_esotic(void) {};
    HD Contract_eq_option_esotic(Equity_prices * eq_prices,
                                  Schedule      * schedule,
                                  double strike_price,
                                  char   contract_type)
        :Contract_eq_option(eq_prices,schedule),
        _contract_type(contract_type),
        _strike_price(strike_price)
        {}
    HD virtual ~Contract_eq_option_esotic(void){}

    //getters and setters
    HD double Get_strike_price()const{return _strike_price;}
    HD char  Get_contract_type()const{return _contract_type;}

    HD void Set_strike_price(const double price){_strike_price = price;}
    HD void Set_contract_type(const char type){_contract_type = type;}

    //functions
  public:
    HD virtual double Pay_off(const Path *path) = 0;
};

#endif