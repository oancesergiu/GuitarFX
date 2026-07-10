import json
from pathlib import Path


class PresetManager:
    def __init__(self, rack, effects, presets_dir="presets"):
        self.rack = rack
        self.effects = effects
        self.presets_dir = Path(presets_dir)

    def available_presets(self):
        return sorted(path.stem for path in self.presets_dir.glob("*.json"))

    def disable_all(self):
        for effect in self.effects.values():
            self.rack.disable(effect)

    def load(self, preset_name):
        preset_path = self.presets_dir / f"{preset_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset '{preset_name}' was not found in {self.presets_dir}"
            )

        with preset_path.open("r", encoding="utf-8") as file:
            preset = json.load(file)

        self.disable_all()

        settings = preset.get("effects", {})

        self._configure_gate(settings.get("gate", {}))
        self._configure_overdrive(settings.get("overdrive", {}))
        self._configure_eq(settings.get("eq", {}))
        self._configure_delay(settings.get("delay", {}))

        print(f"Loaded preset: {preset.get('name', preset_name)}")

    def _set_enabled(self, effect_name, enabled):
        effect = self.effects[effect_name]

        if enabled:
            self.rack.enable(effect)
        else:
            self.rack.disable(effect)

    def _configure_gate(self, settings):
        gate = self.effects["gate"]
        self._set_enabled("gate", settings.get("enabled", False))

        if "threshold" in settings:
            gate.threshold = float(settings["threshold"])

    def _configure_overdrive(self, settings):
        overdrive = self.effects["overdrive"]
        self._set_enabled("overdrive", settings.get("enabled", False))

        if "gain" in settings:
            overdrive.set_gain(settings["gain"])

        if "drive" in settings:
            overdrive.set_drive(settings["drive"])

        if "level" in settings:
            overdrive.set_level(settings["level"])

    def _configure_eq(self, settings):
        eq = self.effects["eq"]
        self._set_enabled("eq", settings.get("enabled", False))

        if "bass_db" in settings:
            eq.set_bass(settings["bass_db"])

        if "mid_db" in settings:
            eq.set_mid(settings["mid_db"])

        if "treble_db" in settings:
            eq.set_treble(settings["treble_db"])

    def _configure_delay(self, settings):
        delay = self.effects["delay"]
        self._set_enabled("delay", settings.get("enabled", False))

        if "delay_ms" in settings:
            delay.set_delay_ms(settings["delay_ms"])

        if "feedback" in settings:
            delay.set_feedback(settings["feedback"])

        if "mix" in settings:
            delay.set_mix(settings["mix"])