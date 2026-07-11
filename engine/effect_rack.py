import numpy as np


class EffectRack:
    """
    Ordered collection of float32 DSP effects.
    """

    def __init__(self):
        self.effects = []
        self.disabled_effects = set()

    def add(self, effect):
        if effect not in self.effects:
            self.effects.append(effect)

    def remove(self, effect):
        if effect in self.effects:
            self.effects.remove(effect)

        self.disabled_effects.discard(effect)

    def enable(self, effect):
        if effect in self.effects:
            self.disabled_effects.discard(effect)

    def disable(self, effect):
        if effect in self.effects:
            self.disabled_effects.add(effect)

    def clear(self):
        self.effects.clear()
        self.disabled_effects.clear()

    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        )

        for effect in self.effects:
            if effect not in self.disabled_effects:
                output = effect.process(output)

                if output.dtype != np.float32:
                    output = output.astype(
                        np.float32,
                        copy=False,
                    )

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)