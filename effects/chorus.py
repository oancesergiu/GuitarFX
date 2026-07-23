import math

import numpy as np


class Chorus:
    """
    CPU-efficient mono chorus using a vectorized fractional delay.

    Designed for normalized float32 audio and block sizes where the
    minimum delay is longer than one processing block.
    """

    def __init__(
        self,
        sample_rate=44100,
        rate_hz=0.8,
        base_delay_ms=18.0,
        depth_ms=4.0,
        mix=0.35,
        max_delay_ms=40.0,
        block_size=256,
    ):
        self.sample_rate = float(sample_rate)
        self.block_size = max(1, int(block_size))

        self.rate_hz = float(rate_hz)
        self.base_delay_ms = float(base_delay_ms)
        self.depth_ms = float(depth_ms)
        self.mix = float(np.clip(mix, 0.0, 1.0))

        self.buffer_size = max(
            2,
            int(
                math.ceil(
                    self.sample_rate
                    * max_delay_ms
                    / 1000.0
                )
            )
            + 2,
        )

        self.buffer = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        self.write_index = 0
        self.phase = 0.0

        self._validate_delay_range()

    def _validate_delay_range(self):
        minimum_delay_ms = (
            self.base_delay_ms - self.depth_ms
        )

        minimum_delay_samples = (
            self.sample_rate
            * minimum_delay_ms
            / 1000.0
        )

        if minimum_delay_ms <= 0.0:
            raise ValueError(
                "Chorus base delay must be greater than depth."
            )

        if minimum_delay_samples <= self.block_size:
            raise ValueError(
                "Minimum chorus delay must exceed the audio block size. "
                "Increase base_delay_ms or reduce depth_ms."
            )

        maximum_delay_samples = (
            self.sample_rate
            * (
                self.base_delay_ms
                + self.depth_ms
            )
            / 1000.0
        )

        if maximum_delay_samples >= self.buffer_size - 1:
            raise ValueError(
                "Chorus delay exceeds the allocated delay buffer."
            )

    def set_rate(self, rate_hz):
        self.rate_hz = float(
            np.clip(rate_hz, 0.05, 10.0)
        )

    def set_depth(self, depth_ms):
        self.depth_ms = max(
            0.0,
            float(depth_ms),
        )
        self._validate_delay_range()

    def set_base_delay(self, delay_ms):
        self.base_delay_ms = max(
            0.1,
            float(delay_ms),
        )
        self._validate_delay_range()

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def reset(self):
        self.buffer.fill(0.0)
        self.write_index = 0
        self.phase = 0.0

    def process_inplace(self, buffer):
        signal = np.asarray(
            buffer,
            dtype=np.float32,
        )

        sample_count = signal.size

        if sample_count == 0:
            return

        if sample_count != self.block_size:
            self.block_size = sample_count
            self._validate_delay_range()

        sample_offsets = np.arange(
            sample_count,
            dtype=np.float64,
        )

        phase_increment = (
            2.0
            * math.pi
            * self.rate_hz
            / self.sample_rate
        )

        phases = (
            self.phase
            + sample_offsets * phase_increment
        )

        # LFO range: -1.0 to +1.0.
        modulation = np.sin(phases)

        delay_ms = (
            self.base_delay_ms
            + self.depth_ms * modulation
        )

        delay_samples = (
            self.sample_rate
            * delay_ms
            / 1000.0
        )

        # Read positions move forward with the block while remaining
        # behind the write head by the modulated fractional delay.
        read_positions = (
            self.write_index
            + sample_offsets
            - delay_samples
            - 1.0
        ) % self.buffer_size

        index0 = np.floor(
            read_positions
        ).astype(np.int64)

        index1 = (
            index0 + 1
        ) % self.buffer_size

        fraction = (
            read_positions - index0
        ).astype(np.float32)

        sample0 = self.buffer[index0]
        sample1 = self.buffer[index1]

        wet = (
            sample0 * (1.0 - fraction)
            + sample1 * fraction
        )

        # Store the current dry block in the circular buffer.
        write_indices = (
            self.write_index
            + np.arange(sample_count)
        ) % self.buffer_size

        self.buffer[write_indices] = signal

        self.write_index = (
            self.write_index + sample_count
        ) % self.buffer_size

        self.phase = (
            self.phase
            + sample_count * phase_increment
        ) % (2.0 * math.pi)

        dry_mix = 1.0 - self.mix

        np.multiply(
            signal,
            dry_mix,
            out=buffer,
        )

        buffer += wet * self.mix

        np.clip(
            buffer,
            -1.0,
            1.0,
            out=buffer,
        )

    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        ).copy()

        self.process_inplace(output)

        return output