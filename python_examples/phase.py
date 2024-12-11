import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

f = 2
N = 100
SR = 100
sine = [np.sin(2 * np.pi * f * n / SR) for n in range(N)]
opp = [np.sin(2 * np.pi * f * n / SR + np.pi) for n in range(N)]
off = [np.sin(2 * np.pi * f * n / SR + np.pi/2) for n in range(N)]
x = np.arange(0, N, 1)
fig, axs = plt.subplots(2)
axs[0].plot(x, sine)
axs[0].plot(x, off)
axs[0].set_ylabel("Décalage de phase")
axs[1].plot(x, sine)
axs[1].plot(x, opp)
axs[1].set_ylabel("Opposition de phase")
plt.show()