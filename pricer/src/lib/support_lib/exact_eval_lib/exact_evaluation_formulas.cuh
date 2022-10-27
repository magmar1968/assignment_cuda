#ifndef __EXEVFORMULAS__
#define __EXEVFORMULAS__

#include <math.h>
#include "../../contract_option_lib/contract_eq_option_esotic/contract_eq_option_esotic.cuh"
#include "../../contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"

namespace prcr
{
	__host__ double Evaluate_forward(const double S_0, const double T, const double rate);
	__host__ double Evaluate_vanilla(const Equity_prices* starting_point, const Contract_eq_option_vanilla* contract);
	__host__ double Evaluate_vanilla(const char contract_type, const double sigma, const double r, const double S_0, const double T, const double E);
	__host__ double Evaluate_corridor(Equity_prices* starting_point, const Contract_eq_option_esotic* contract);
	__host__ double Evaluate_corridor(double rate, double sigam, int m, double T, double K, double B);
	__host__ double Compute_P_corridor_single_step(double rate, double sigma, int m, double T, double B);



	__host__ int Factorial(int x)
	{
		return tgamma(x + 1);
	}

	__host__ double Evaluate_forward(const double S_0, const double T, const double rate)
	{
		return S_0 * exp(rate * T);
	}

	__host__ double Evaluate_vanilla(const Equity_prices* starting_point, const Contract_eq_option_vanilla* contract)
	{
		char ctype = contract->Get_contract_type();
		double sigma = starting_point->Get_eq_description()->Get_vol_surface();
		double r = starting_point->Get_eq_description()->Get_yc();
		double S_0 = starting_point->Get_price();
		Schedule* sch = contract->Get_schedule();
		double T = sch->Get_t(sch->Get_dim() - 1);
		double E = contract->Get_strike_price();

		return Evaluate_vanilla(ctype, sigma, r, S_0, T, E);
	}

	__host__ double Evaluate_vanilla(const char type, const double sigma, const double r, const double S_0, const double T, const double E)
	{

		if (type == 'C')
		{
			
			if (E > 0 && S_0 > 0)  //if false, we can't use log
			{
				double d1 = (log(S_0 / E) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T));
				double d2 = (log(S_0 / E) + (r - sigma * sigma / 2) * T) / (sigma * sqrt(T));

				double N1 = 0.5 * (1 + erf(d1 / sqrt(2)));
				double N2 = 0.5 * (1 + erf(d2 / sqrt(2)));


				return S_0 * N1 * exp(r * T) - E * N2;
			}
			if (E == 0 && S_0 > 0)
				return Evaluate_forward(S_0, T, r);  //return forward (useful when E = 0, vanilla becomes forward)
			else return -1;
			
		}
		if (type == 'P')
		{
			if (E > 0 && S_0 > 0) //otherwise we can't use log
			{
				double d1 = (log(S_0 / E) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T));
				double d2 = (log(S_0 / E) + (r - sigma * sigma / 2) * T) / (sigma * sqrt(T));

				double N1 = 0.5 * (1 + erf(-d1 / sqrt(2)));
				double N2 = 0.5 * (1 + erf(-d2 / sqrt(2)));

				return E * N2 - S_0 * N1 * exp(r * T);
			}
			else
				return -1;
		}
		else return -1;
		
			

	}


	__host__ double Evaluate_corridor(Equity_prices* starting_point, const Contract_eq_option_esotic* contract)
	{
		char ctype = contract->Get_contract_type();
		if (ctype == 'C')
		{
			K = contract->Get_K();
			B = contract->Get_B();
			double sigma = starting_point->Get_eq_description()->Get_vol_surface();
			double r = starting_point->Get_eq_description()->Get_yc();
			Schedule* sch = contract->Get_schedule();
			double T = sch->Get_t(sch->Get_dim() - 1);
			int m = sch->Get_dim();
			return Evaluate_corridor(r,sigma,m,T,K,B)
		}
		else
		{
			return -1;
		}

	}


	__host__ double Evaluate_corridor(double rate, double sigma, int m, double T, double K, double B)
	{
		estr = min(floor(m * K), m);
		expected_value = 0;
		P = Compute_P_corridor_single_step(rate, sigma, m, T, B);
		for (int n = 0; n < estr; n++)
		{
			expected_value += (K - double(m) / double(n)) * double(factorial(m)) / (double(factorial(m-n) * factorilal(n)) * pow(P, n) * pow(P, m - n);
		}
		return expected_value;
	}


	__host__ double Compute_P_corridor_single_step(double rate, double sigma, int m, double T, double B)
	{
		double deltat = T / double(m);
		double sup = -sqrt(deltat) * (rate - sigma * sigma/2) + B;
		double inf = -sqrt(deltat) * (rate - sigma * sigma/2) - B;
		return 0.5*( - erf(inf / sqrt(2)) + erf(sup / sqrt(2)));
	}
}
#endif