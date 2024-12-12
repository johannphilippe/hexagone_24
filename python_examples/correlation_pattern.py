import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

N = 100
s1 = np.zeros(N)
s2 = np.zeros(N)

for i in range(10):
    s1[i+15] = 1
    s2[i+65] = 1

corr = np.ndarray(N)

for k in range(N):
    corr[k] = 0
    for i in range(N):
        j = i + k
        if(j < N):
            corr[k] += s1[i] * s2[j] 


x = np.arange(0, N, 1)
fig, axs = plt.subplots(3)
axs[0].plot(x, s1)
axs[0].set_ylabel("Signal 1")
axs[1].plot(x, s2)
axs[1].set_ylabel("Signal 2")
axs[2].plot(x, corr)
axs[2].set_ylabel("Correlation")
plt.show()
