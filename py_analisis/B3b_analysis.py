import numpy as np
import matplotlib.pyplot as plt 

pathoutfile = "../pricer/data/outfile_exam/"

data_k100 = np.genfromtxt(pathoutfile + "outfile_B3b_K1.txt"  , delimiter = ',', comments='#')
data_k050 = np.genfromtxt(pathoutfile + "outfile_B3b_K05.txt"  , delimiter = ',', comments='#')
data_k075 = np.genfromtxt(pathoutfile + "outfile_B3b_K075.txt", delimiter = ',', comments='#')

B_values = data_k050[:,0]
scale = 1.2

plt.plot(B_values,data_k100[:,1],"-o",label = "K = 1"   )
plt.plot(B_values,data_k075[:,1],"-o",label = "K = 0.75")
plt.plot(B_values,data_k050[:,1],"-o",label = "K = 0.50")

plt.ylabel(r"Pay off",size=12*scale)
plt.xlabel(r"B",size=12*scale)
plt.xticks(size = 10*scale)
plt.yticks(size = 10*scale)
plt.tick_params(axis="both", direction='in', length=6*scale,width=1*scale)
plt.legend( prop={'size': 10*scale})
plt.show()

plt.plot(B_values,data_k100[:,2],"-o",label = "K = 1   ")
plt.plot(B_values,data_k075[:,2],"-o",label = "K = 0.75")
plt.plot(B_values,data_k050[:,2],"-o",label = "K = 0.50")

plt.ylabel(r"MC error",size=12*scale)
plt.xlabel(r"B",size=12*scale)
plt.xticks(size = 10*scale)
plt.yticks(size = 10*scale)
plt.tick_params(axis="both", direction='in', length=6*scale,width=1*scale)
plt.legend( prop={'size': 10*scale})
plt.show()

