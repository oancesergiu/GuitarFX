import numpy as np


class EffectRack:
    """
    Ordered collection of normalized float32 DSP effects.

    Effects may implement either:

        process_inplace(buffer)

    or:

        process(buffer) -> buffer
    """

    def __init__(self, benchmark=None):
        self.effects = []
        self.disabled_effects = set()
        self.benchmark = benchmark

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

    def _process_effect(self, effect, buffer):
        process_inplace = getattr(
            effect,
            "process_inplace",
            None,
        )

        if callable(process_inplace):
            process_inplace(buffer)
            return buffer

        result = effect.process(buffer)

        result = np.asarray(
            result,
            dtype=np.float32,
        )

        if result is not buffer:
            np.copyto(buffer, result)

        return buffer

    def process(self, signal):
        buffer = np.asarray(
            signal,
            dtype=np.float32,
        )

        if not buffer.flags.writeable:
            buffer = buffer.copy()

        for effect in self.effects:
            if effect in self.disabled_effects:
                continue

            if self.benchmark is None:
                self._process_effect(effect, buffer)
            else:
                effect_name = effect.__class__.__name__

                self.benchmark.measure(
                    effect_name,
                    self._process_effect,
                    effect,
                    buffer,
                )

        np.clip(
            buffer,
            -1.0,
            1.0,
            out=buffer,
        )

        return buffer