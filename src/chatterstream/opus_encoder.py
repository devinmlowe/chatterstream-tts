"""OGG/Opus encoder: PCM int16 bytes to OGG/Opus stream using PyAV."""

from __future__ import annotations

import io

import av
import numpy as np


class OpusEncoder:
    """Encodes PCM int16 mono audio to OGG/Opus using PyAV.

    Designed for streaming: encode() is called per chunk, finalize() flushes
    the encoder and closes the OGG stream. reset() prepares for a new stream.

    Opus requires 48kHz input, so 24kHz PCM is resampled transparently.
    """

    def __init__(self, input_sample_rate: int = 24000, bitrate: int = 64000):
        self._input_sr = input_sample_rate
        self._bitrate = bitrate
        self._pts = 0
        self._buf: io.BytesIO | None = None
        self._output: av.container.OutputContainer | None = None
        self._stream: av.audio.AudioStream | None = None
        self._resampler: av.AudioResampler | None = None
        self._open()

    def _open(self) -> None:
        """Create a fresh OGG/Opus container and encoder."""
        self._buf = io.BytesIO()
        self._output = av.open(self._buf, mode="w", format="ogg")
        self._stream = self._output.add_stream("libopus", rate=48000)
        self._stream.bit_rate = self._bitrate
        self._stream.layout = "mono"
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=48000
        )
        self._pts = 0
        self._last_read_pos = 0

    def encode(self, pcm_bytes: bytes) -> bytes:
        """Encode a chunk of PCM int16 mono audio. Returns OGG pages produced."""
        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            return b""

        # Create input frame at source sample rate
        frame = av.AudioFrame(format="s16", layout="mono", samples=n_samples)
        frame.planes[0].update(pcm_bytes)
        frame.sample_rate = self._input_sr
        frame.pts = self._pts

        # Resample to 48kHz if needed
        if self._input_sr != 48000:
            resampled = self._resampler.resample(frame)
        else:
            resampled = [frame]

        for rf in resampled:
            rf.pts = self._pts
            for packet in self._stream.encode(rf):
                self._output.mux(packet)
            self._pts += rf.samples

        # Read newly written bytes
        return self._drain()

    def finalize(self) -> bytes:
        """Flush encoder buffer and close the OGG stream."""
        if self._stream is None:
            return b""

        # Flush encoder
        for packet in self._stream.encode(None):
            self._output.mux(packet)

        self._output.close()
        result = self._drain()
        self._output = None
        self._stream = None
        return result

    def reset(self) -> None:
        """Reset for a new OGG/Opus stream."""
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
        self._open()

    def close(self) -> None:
        """Close the encoder and release resources."""
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
            self._output = None
            self._stream = None

    def __del__(self) -> None:
        self.close()

    def _drain(self) -> bytes:
        """Read any new bytes written to the buffer since last drain."""
        pos = self._buf.tell()
        if pos <= self._last_read_pos:
            return b""
        self._buf.seek(self._last_read_pos)
        data = self._buf.read(pos - self._last_read_pos)
        self._buf.seek(pos)
        self._last_read_pos = pos
        return data
