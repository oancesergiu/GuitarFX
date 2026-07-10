class PresetManager:
    def __init__(self, rack, effects):
        self.rack = rack
        self.effects = effects

    def disable_all(self):
        for effect in self.effects.values():
            self.rack.disable(effect)

    def load(self, preset_name):
        self.disable_all()

        if preset_name == "clean":
            self.rack.enable(self.effects["gate"])
            self.rack.enable(self.effects["eq"])

        elif preset_name == "rock":
            self.rack.enable(self.effects["gate"])
            self.rack.enable(self.effects["overdrive"])
            self.rack.enable(self.effects["eq"])

        elif preset_name == "lead":
            self.rack.enable(self.effects["gate"])
            self.rack.enable(self.effects["overdrive"])
            self.rack.enable(self.effects["eq"])
            self.rack.enable(self.effects["delay"])

        else:
            raise ValueError(f"Unknown preset: {preset_name}")