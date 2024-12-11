import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

f = 2
N = 100
SR = 100
sine = [np.sin(2 * np.pi * f * n / SR) for n in range(N)]
noise = [random.random() - 0.5 * 2 for n in range(N)]
x = np.arange(0, N, 1)
fig, axs = plt.subplots(2)
axs[0].plot(x, sine)
axs[1].plot(x, noise)
plt.show()