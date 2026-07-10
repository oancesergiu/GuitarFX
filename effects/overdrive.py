from effects.distortion import soft_clip


class Overdrive:
    def __init__(self, gain=3.5, drive=16000.0, level=22000.0):
        self.gain = gain
        self.drive = drive
        self.level = level

    def set_gain(self, gain):
        self.gain = float(gain)

    def set_drive(self, drive):
        self.drive = float(drive)

    def set_level(self, level):
        self.level = float(level)

    def process(self, signal):
        return soft_clip(
            signal,
            gain=self.gain,
            drive=self.drive,
            level=self.level,
        )