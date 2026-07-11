from effects.noise_gate import noise_gate


class NoiseGate:
    def __init__(self, threshold=0.008):
        self.threshold = float(threshold)

    def set_threshold(self, threshold):
        self.threshold = max(0.0, float(threshold))

    def process(self, signal):
        return noise_gate(
            signal,
            threshold=self.threshold,
        )