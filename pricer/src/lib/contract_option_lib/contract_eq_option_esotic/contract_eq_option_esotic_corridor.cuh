#ifndef __CONTRACT_EQ_OPTION_CORRIDOR__
#define __CONTRACT_EQ_OPTION_CORRIDOR__

#include "contract_eq_option_esotic.cuh"


namespace prcr
{

	class Contract_eq_option_exotic_corridor: public Contract_eq_option_esotic
	{
	public:
		HD Contract_eq_option_exotic_corridor() {};
		HD Contract_eq_option_exotic_corridor(  Equity_prices* eq_prices,
												Schedule* schedule,
												double strike_price,
												char contract_type,
												double B,
												double N,
												double K )
			:Contract_eq_option_esotic(eq_prices, schedule, strike_price, contract_type),
			_Noz(N), _B(B), _K(K)
		{
			_sigma = eq_prices->Get_eq_description()->Get_vol_surface();
			_delta_t = schedule->Get_t(1) - schedule->Get_t(0);  //tempi di schedule equispaziati per ipotesi
			_steps = schedule->Get_dim();
		}

		HD void Set_B(double B) { _B = B; }
		HD void Set_N(double N) { _Noz = N; }
		HD void Set_K(double K) { _K = K; }
		HD double Get_B(void) { return _B; }
		HD double Get_N(void) { return _Noz; }
		HD double Get_K(void) { return _K; }

		HD ~Contract_eq_option_exotic_corridor() {};
		HD double Pay_off(const Path* path);

	private:
		double _Noz;
		double _K;
		double _B;
		double _sigma;
		double _delta_t;
		double _steps;

		HD bool Evaluate_log_return(size_t i, const Path* path);
	};


	HD double Contract_eq_option_exotic_corridor::Pay_off(const Path* path)
	{
		int P = 0;
		for (size_t i = 0; i < _steps-1; i++)
		{
			if (Evaluate_log_return(i, path))
				P++;
		}

		if (_contract_type == 'P')
			return max((_Noz * (_K - P / (_steps - 1))), 0.);   //o steps e basta?
		else
			return 0; //non abbiamo formula per call performance corridor


	}



HD bool Contract_eq_option_exotic_corridor::Evaluate_log_return(size_t i, const Path* path)
{
	double S_a = path->Get_equity_prices(i +1);
	double S_b = path->Get_equity_prices(i);
	if (abs((log(S_a / S_b)) / sqrt(_delta_t)) < _B * _sigma)
		return 1;
	else
		return 0;
}

}


#endif
