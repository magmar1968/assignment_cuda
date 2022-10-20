import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv("../pricer/data/outfile_B3a_m_values.csv")

m_values = df["m"].tolist()
exact_vals = df["exact_result"].to_list()
approx_vals = df["approx_result"].to_list()


plt.plot(m_values,approx_vals,label = r"pay off approx")
plt.plot(m_values,exact_vals, label = r"pay off exact ")
# plt.yscale('log')


plt.legend()
# plt.savefig("../plots/B3a_plot.png",format="png")
plt.show()