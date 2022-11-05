import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("../pricer/data/outfile_E2_binom.csv")

m_values = df["m"].to_numpy()
p_off_l  = df["p_off_lognorm"].to_numpy()
p_off_b  = df["p_off_binom"].to_numpy()
mc_err_l = df["MC_err_lognorm"].to_numpy()
mc_err_b = df["MC_error_binom"].to_numpy()


plt.plot(m_values,p_off_l,'-o',color = "blue",label=r"lognorm")
plt.fill_between(m_values,p_off_l + mc_err_l, p_off_l-mc_err_l, interpolate=False,alpha = 0.4)

plt.plot(m_values,p_off_b,"-o",color = "coral", label =r"binomial" )
plt.fill_between(m_values,p_off_b - mc_err_b, p_off_b+mc_err_b,interpolate=False,alpha = 0.3)

plt.xlabel("m")
plt.ylabel("pay_off")


plt.legend()
plt.show()
