import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv("../pricer/data/outfile_B3a_m_values.csv")

m_values = df["m"].tolist()
exact_vals = df["p_off_exact"].to_list()
approx_vals = df["p_off_approx"].to_list()


plt.plot(m_values,approx_vals,label = r"pay off approx")
plt.plot(m_values,exact_vals, label = r"pay off exact ")


plt.legend()
plt.show()