import numpy as np

def apply_gain(signal, gain=1.0):
    signal = signal.astype(np.float32)
    signal *= gain
    signal = np.clip(signal, -32768, 32767)
    return signal.astype(np.int16)