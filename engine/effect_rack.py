class EffectRack:
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
        self.disabled_effects.discard(effect)

    def disable(self, effect):
        if effect in self.effects:
            self.disabled_effects.add(effect)

    def clear(self):
        self.effects.clear()
        self.disabled_effects.clear()

    def process(self, signal):
        for effect in self.effects:
            if effect not in self.disabled_effects:
                signal = effect.process(signal)

        return signal