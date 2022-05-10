import matplotlib.pyplot as plt
import csv
import numpy as np


# data_cpu = np.genfromtxt("./cpu_performance.txt", delimiter = ',')
data_gpu = np.genfromtxt("./gpu_performance.txt", delimiter = ',')


# plot data


# plt.plot(data_cpu[:,0], data_cpu[:,1] , label = 'cpu')
plt.plot(data_gpu[:,0], data_gpu[:,1] , label = 'gpu')


plt.xlabel("array lenght")
plt.ylabel(r'# $\mu$ s for sum')
plt.legend()
plt.show()