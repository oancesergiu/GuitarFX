import numpy as np

def soft_clip(signal, gain=4.0, drive=18000.0, level=28000.0):
    signal = signal.astype(np.float32)
    signal *= gain
    signal = np.tanh(signal / drive) * level
    return signal.astype(np.int16)