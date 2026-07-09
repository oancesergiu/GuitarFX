class EffectRack:
    def __init__(self):
        self.effects = []

    def add(self, effect):
        self.effects.append(effect)

    def remove(self, effect):
        if effect in self.effects:
            self.effects.remove(effect)

    def clear(self):
        self.effects.clear()

    def process(self, signal):
        for effect in self.effects:
            signal = effect.process(signal)

        return signal