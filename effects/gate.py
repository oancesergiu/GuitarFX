from effects.noise_gate import noise_gate


class NoiseGate:
    def __init__(self, threshold=250):
        self.threshold = threshold

    def process(self, signal):
        return noise_gate(signal, threshold=self.threshold)