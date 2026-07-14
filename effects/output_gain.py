import numpy as np


class OutputGain:
    """
    Final preset-controlled gain stage for normalized float32 audio.
    Place immediately before the limiter.
    """

    def __init__(self, gain_db=0.0):
        self.set_gain_db(gain_db)

    def set_gain_db(self, gain_db):
        self.gain_db = float(np.clip(gain_db, -24.0, 12.0))
        self.gain_linear = 10.0 ** (self.gain_db / 20.0)

    def process_inplace(self, buffer):
        buffer *= self.gain_linear

    def process(self, signal):
        output = np.asarray(signal, dtype=np.float32).copy()
        self.process_inplace(output)
        return output