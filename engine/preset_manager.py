import json
from pathlib import Path
from engine.amp_factory import AmpFactory

class PresetManager:
    def __init__(self, rack, effects, presets_dir="presets"):
        self.rack = rack
        self.effects = effects
        self.presets_dir = Path(presets_dir)

    def available_presets(self):
        return sorted(
            path.stem
            for path in self.presets_dir.glob("*.json")
        )

    def disable_all(self):
        for effect in self.effects.values():
            self.rack.disable(effect)

    def load(self, preset_name):
        preset_path = self.presets_dir / f"{preset_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset '{preset_name}' was not found in "
                f"{self.presets_dir}"
            )

        with preset_path.open("r", encoding="utf-8") as file:
            preset = json.load(file)

        self.disable_all()

        settings = preset.get("effects", {})

        self._configure_gate(
            settings.get("gate", {})
        )

        self._configure_overdrive(
            settings.get("overdrive", {})
        )

        self._configure_amp_model(
            settings.get("amp_model", {})
        )

        self._configure_eq(
            settings.get("eq", {})
        )

        self._configure_auto_wah(
            settings.get("auto_wah", {})
        )

        self._configure_cabinet(
            settings.get("cabinet", {})
        )

        self._configure_delay(
            settings.get("delay", {})
        )

        self._configure_output_gain(
           settings.get("output_gain", {})
        )

        print(
            f"Loaded preset: "
            f"{preset.get('name', preset_name)}"
        )

    def _set_enabled(self, effect_name, enabled):
        effect = self.effects[effect_name]

        if enabled:
            self.rack.enable(effect)
        else:
            self.rack.disable(effect)

    def _configure_gate(self, settings):
        gate = self.effects["gate"]

        self._set_enabled(
            "gate",
            settings.get("enabled", False),
        )

        if "threshold" in settings:
            gate.threshold = float(
                settings["threshold"]
            )

    def _configure_overdrive(self, settings):
        overdrive = self.effects["overdrive"]

        self._set_enabled(
            "overdrive",
            settings.get("enabled", False),
        )

        if "gain" in settings:
            overdrive.set_gain(
                settings["gain"]
            )

        if "drive" in settings:
            overdrive.set_drive(
                settings["drive"]
            )

        if "level" in settings:
            overdrive.set_level(
                settings["level"]
            )

        if "tone" in settings:
            overdrive.set_tone(
                settings["tone"]
            )

    def _configure_eq(self, settings):
        eq = self.effects["eq"]

        self._set_enabled(
            "eq",
            settings.get("enabled", False),
        )

        if "bass_db" in settings:
            eq.set_bass(
                settings["bass_db"]
            )

        if "mid_db" in settings:
            eq.set_mid(
                settings["mid_db"]
            )

        if "treble_db" in settings:
            eq.set_treble(
                settings["treble_db"]
            )

    def _configure_auto_wah(self, settings):
        auto_wah = self.effects["auto_wah"]

        self._set_enabled(
            "auto_wah",
            settings.get("enabled", False),
        )

        if "rate_hz" in settings:
            auto_wah.set_rate(
                settings["rate_hz"]
            )

        if "mix" in settings:
            auto_wah.set_mix(
                settings["mix"]
            )

        if (
            "min_frequency" in settings
            and "max_frequency" in settings
        ):
            auto_wah.set_frequency_range(
                settings["min_frequency"],
                settings["max_frequency"],
            )

        if "resonance_q" in settings:
            auto_wah.set_resonance(
                settings["resonance_q"]
            )

    def _configure_cabinet(self, settings):
        cabinet = self.effects["cabinet"]

        self._set_enabled(
            "cabinet",
            settings.get("enabled", False),
        )

        if "output_level" in settings:
            cabinet.output_level = float(
                settings["output_level"]
            )
        if "ir" in settings:
            cabinet.load_ir(
            settings["ir"]
        )
        

    def _configure_delay(self, settings):
        delay = self.effects["delay"]

        self._set_enabled(
            "delay",
            settings.get("enabled", False),
        )

        if "delay_ms" in settings:
            delay.set_delay_ms(
                settings["delay_ms"]
            )

        if "feedback" in settings:
            delay.set_feedback(
                settings["feedback"]
            )

        if "mix" in settings:
            delay.set_mix(
                settings["mix"]
            )
    def _configure_output_gain(self, settings):
        output_gain = self.effects["output_gain"]

        self._set_enabled(
            "output_gain",
            settings.get("enabled", True),
        )

        if "gain_db" in settings:
            output_gain.set_gain_db(
                settings["gain_db"]
            )

    def _configure_amp_model(self, settings):
        amp_model = self.effects["amp_model"]

        self._set_enabled(
            "amp_model",
            settings.get("enabled", True),
        )

        if "model" in settings:
            AmpFactory.configure(
                amp_model,
                settings["model"],
            )

        if "preamp_gain" in settings:
            amp_model.set_preamp_gain(
                settings["preamp_gain"]
            )

        if "second_stage_gain" in settings:
            amp_model.set_second_stage_gain(
                settings["second_stage_gain"]
            )

        if "bass" in settings:
            amp_model.set_bass(
                settings["bass"]
            )

        if "middle" in settings:
            amp_model.set_middle(
                settings["middle"]
            )

        if "treble" in settings:
            amp_model.set_treble(
                settings["treble"]
            )

        if "presence" in settings:
            amp_model.set_presence(
                settings["presence"]
            )

        if "resonance" in settings:
            amp_model.set_resonance(
                settings["resonance"]
            )

        if "sag" in settings:
            amp_model.set_sag(
                settings["sag"]
            )

        if "master" in settings:
            amp_model.set_master(
                settings["master"]
            )

