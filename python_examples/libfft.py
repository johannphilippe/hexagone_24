
import numpy as np
from math import *

def magnitude(fft_sig):
    l = len(fft_sig[0])
    mag = np.zeros(l)
    for k in range(l):
        mag[k] = sqrt( pow(fft_sig[0][k], 2) + pow(fft_sig[1][k], 2) )
    return mag

def fft_r2c(sig):
    fft_size = len(sig) 
    result = np.ndarray(shape=(2, fft_size))
    # For each FFT bin (each frequency we try)
    for k in range(fft_size):
        result[0][k] = 0
        result[1][k] = 0
        # Iterate over samples to determine how much signal samples match our sines and cosines
        for n in range(fft_size):
            # Here k is frequency and n is phase
            nn = -2 * pi / fft_size * k * n 
            result[0][k] += cos(nn) * sig[n] 
            result[1][k] += sin(nn) * sig[n]
    return result

def complex_conjugate(complex_sig):
    l = len(complex_sig[0])
    conj = complex_sig.copy()
    for n in range(l):
        complex_sig[0][n] = complex_sig[0][n] - complex_sig[1][n]

# IFFT can be considered as the same operation as FFT, by applying complex conjugate before and after FFT
def ifft_c2r(fft_sig):
    fft_size = len(fft_sig[0])
    complex_conjugate(fft_sig)

    ifft_sig = np.ndarray(shape=(2, fft_size))
    for k in range(fft_size):
        ifft_sig[0][k] = 0
        ifft_sig[1][k] = 0
        for n in range(fft_size): 
            nn = -2 * pi / fft_size * k * n 
            ifft_sig[0][k] += cos(nn)  * fft_sig[0][n] 
            ifft_sig[1][k] += sin(nn)  * fft_sig[0][n]

        ifft_sig[0][k] = (ifft_sig[0][k] - ifft_sig[1][k]) / fft_size

    real = ifft_sig[0].copy()
    return real


def psd(fft_sig, sr):
    l = len(fft_sig[0])
    power = np.ndarray(l)
    mags = magnitude(fft_sig)

    psd = np.ndarray(l)
    for n in range(l):
        # PSD can be described as the square of the magnitude
        res = (2 / ( sr * l)) * np.abs(complex(fft_sig[0][n], fft_sig[1][n])) ** 2
        psd[n] = res
    return psd
    

def fft_conv(sig, ir):
    l = len(sig[0])
    res = np.ndarray(shape=(2, l) )
    for n in range(l):
        # convolution in time domain becomes simple product in frequency domain
        r = complex(sig[0][n], sig[1][n]) * complex(ir[0][n], ir[1][n])
        res[0][n] = r.real
        res[1][n] = r.imag
    return res

