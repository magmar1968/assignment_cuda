import pandas as pd 
import matplotlib.pyplot as plt

pathoutfile = "../pricer/data/outfile_exam/"
df = pd.read_csv(pathoutfile + "outfile_B3a.csv")

m_values = df["m"].to_numpy()
exact_vals = df["exact_result"].to_numpy()
exact_err  = df["exact_error"].to_numpy()
approx_vals = df["approx_result"].to_numpy()
approx_err  = df["approx_erro"].to_numpy()
scale = 1.2

plt.plot(m_values,approx_vals,"-o",label = r"pay off approx")
plt.plot(m_values,exact_vals, "-o",label = r"pay off exact ")
plt.ylabel(r"Pay off", size = 12*scale)
plt.xlabel(r"m",       size = 12*scale)
plt.xticks(size = 10*scale)
plt.yticks(size = 10*scale)
plt.tick_params(axis="both", direction='in', length=6*scale,width=1*scale)
plt.legend( prop={'size': 10*scale})

plt.show()