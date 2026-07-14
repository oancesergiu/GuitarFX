import numpy as np

from engine.dsp.delay_line import DelayLine


class Delay:
    """
    Feedback delay for normalized float32 audio.

    Uses vectorized block processing when the delay is at least
    as long as the current audio block.
    """

    def __init__(
        self,
        sample_rate=44100,
        delay_ms=350.0,
        feedback=0.35,
        mix=0.30,
        max_delay_ms=2000.0,
    ):
        self.sample_rate = int(sample_rate)

        max_delay_samples = max(
            2,
            int(self.sample_rate * max_delay_ms / 1000.0),
        )

        self.delay_line = DelayLine(max_delay_samples)

        self.feedback = 0.0
        self.mix = 0.0

        self.set_feedback(feedback)
        self.set_mix(mix)
        self.set_delay_ms(delay_ms)

    def reset(self):
        self.delay_line.reset()

    def set_delay_ms(self, delay_ms):
        self.delay_ms = max(0.1, float(delay_ms))

        self.delay_samples = max(
            1,
            int(self.sample_rate * self.delay_ms / 1000.0),
        )

        self.delay_samples = min(
            self.delay_samples,
            self.delay_line.size - 1,
        )

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, 0.0, 0.95)
        )

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def _process_sample_by_sample(self, signal):
        """
        Fallback for extremely short delays shorter than one block.
        """
        output = np.empty_like(signal)
        dry_mix = 1.0 - self.mix

        for index, sample in enumerate(signal):
            delayed = self.delay_line.read(self.delay_samples)

            output[index] = (
                sample * dry_mix
                + delayed * self.mix
            )

            self.delay_line.write(
                sample + delayed * self.feedback
            )

        return output

    def _process_block(self, signal):
        """
        Fast vectorized path used when delay_samples >= block size.
        """
        block_size = signal.size
        buffer = self.delay_line.buffer
        buffer_size = self.delay_line.size
        write_index = self.delay_line.write_index

        read_indices = (
            write_index
            - self.delay_samples
            - 1
            + np.arange(block_size)
        ) % buffer_size

        delayed = buffer[read_indices].copy()

        output = (
            signal * (1.0 - self.mix)
            + delayed * self.mix
        )

        values_to_write = (
            signal
            + delayed * self.feedback
        )

        write_indices = (
            write_index
            + np.arange(block_size)
        ) % buffer_size

        buffer[write_indices] = values_to_write

        self.delay_line.write_index = (
            write_index + block_size
        ) % buffer_size

        return output

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        if self.delay_samples >= signal.size:
            output = self._process_block(signal)
        else:
            output = self._process_sample_by_sample(signal)

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)