#ifndef __EXEVFORMULAS__
#define __EXEVFORMULAS__

#include <math.h>
#include "../../contract_option_lib/contract_eq_option_esotic/contract_eq_option_esotic.cuh"
#include "../../contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"

namespace prcr
{




	double Evaluate_forward(const double S_0, const double T, const double rate);
	double Evaluate_vanilla(const Equity_prices* starting_point, const Contract_eq_option_vanilla* contract);
	double Evaluate_vanilla(const char contract_type, const double sigma, const double r, const double S_0, const double T, const double E);
	double Evaluate_corridor(Equity_prices* starting_point, const Contract_eq_option_esotic* contract);



	double Evaluate_forward(const double S_0, const double T, const double rate)
	{
		return S_0 * exp(rate * T);
	}

	double Evaluate_vanilla(const Equity_prices* starting_point, const Contract_eq_option_vanilla* contract)
	{
		char ctype = contract->Get_contract_type();
		double sigma = starting_point->Get_eq_description()->Get_vol_surface();
		double r = starting_point->Get_eq_description()->Get_yc();
		double S_0 = starting_point->Get_price();
		Schedule* sch = contract->Get_schedule();
		double T = sch->Get_t(sch->Get_dim() - 1);
		double E = contract->Get_strike_price();

		if (E > 0)
			return Evaluate_vanilla(ctype, sigma, r, S_0, T, E);
		else
			return Evaluate_forward(S_0, T, r);
	}

	double Evaluate_vanilla(const char type, const double sigma, const double r, const double S_0, const double T, const double E)
	{
		if (E > 0 && S_0 > 0)
		{
			double d1 = (log(S_0 / E) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T));
			double d2 = (log(S_0 / E) + (r - sigma * sigma / 2) * T) / (sigma * sqrt(T));

			if (type == 'C')
			{
				double N1 = 0.5 / sqrt(2 * M_PI) * (1 + erf(d1 / sqrt(2)));
				double N2 = 0.5 / sqrt(2 * M_PI) * (1 + erf(d2 / sqrt(2)));

				return S_0 * N1 * exp(r * T) - E * N2;
			}
			if (type == 'P')
			{
				double N1 = 0.5 / sqrt(2 * M_PI) * (1 + erf(-d1 / sqrt(2)));
				double N2 = 0.5 / sqrt(2 * M_PI) * (1 + erf(-d2 / sqrt(2)));

				return E * N2 - S_0 * N1 * exp(r * T);
			}
			else return -1;
		}
		else
		{
			return Evaluate_forward(S_0, r, T);
		}
			

	}


	double Evaluate_corridor(Equity_prices* starting_point, const Contract_eq_option_esotic* contract)
	{
		return 0;
	}
}
#endif