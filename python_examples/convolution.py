import libfft
import windows

import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

SR = 48000
freq = 100
N = 1024

sig = np.ndarray(N)
ir = np.ndarray(N)

for n in range(N):
    sig[n] = random.random() - 0.5
    ir[n] = sin(2 * pi * (n/SR) * freq) * 0.5 + sin(2 * pi * (n/SR) * freq * 2) * 0.3 *  pow(sin(2 * pi * (n/SR) * freq * 4), 8) * 0.2

def manual_convolution(x, h):
    x_len = len(x)
    h_len = len(h)
    y_len = x_len + h_len - 1
    y = np.zeros(y_len)
    
    # Compute convolution
    for n in range(y_len):
        for k in range(h_len):
            if 0 <= n - k < x_len:
                y[n] += x[n - k] * h[k]
    return y


conv = manual_convolution(sig , ir)
npconv = np.convolve(sig, ir, mode='full')

# Calculate where the full overlap is 

fft_sig = libfft.fft_r2c(sig)
fft_ir = libfft.fft_r2c(ir)
fft_conv = libfft.fft_conv(fft_sig, fft_ir)
reconstructed = libfft.ifft_c2r(fft_conv)
rlen = len(reconstructed)

x = np.arange(0, N, 1)
xc = np.arange(0, len(conv), 1)
xrl = np.arange(0, rlen, 1)

fig, axs = plt.subplots(5)
axs[0].plot(x, sig)
axs[0].set_ylabel("Sig 1")
axs[1].plot(x, ir)
axs[1].set_ylabel("Sig 2 - impulse response")
axs[2].plot(x, conv[1023:2048])
axs[2].set_ylabel("Manual Convolution")
axs[3].plot(x, npconv[1023:2048])
axs[3].set_ylabel("Numpy convolution")
axs[4].plot(xrl, reconstructed)
axs[4].set_ylabel("FFT Convolution")
plt.show()
