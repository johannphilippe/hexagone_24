import numpy as np
from math import *

def hanning(sig):
    l = len(sig)
    for n in range(l):
        sig[n] *= (0.5 * (1 - cos(2 * pi * n / (l - 1))))