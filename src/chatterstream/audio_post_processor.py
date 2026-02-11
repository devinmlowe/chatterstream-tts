"""Audio post-processor: tensor to PCM int16 bytes with crossfade."""

from __future__ import annotations

from typing import Optional

import torch

from .interfaces import AudioPostProcessorBase

_DEFAULT_SR = 24000
# Fade-in to suppress initial click artifacts (first chunk only)
_FADE_SILENCE_MS = 5
_FADE_RAMP_MS = 10
# Crossfade duration at chunk boundaries to smooth seams
_CROSSFADE_MS = 20


def _build_fade_in(sample_rate: int) -> torch.Tensor:
    """Build a fade-in envelope: silence then cosine ramp."""
    n_silence = sample_rate * _FADE_SILENCE_MS // 1000
    n_ramp = sample_rate * _FADE_RAMP_MS // 1000
    fade = torch.zeros(n_silence + n_ramp)
    fade[n_silence:] = (torch.cos(torch.linspace(torch.pi, 0, n_ramp)) + 1) / 2
    return fade


class AudioPostProcessor(AudioPostProcessorBase):
    """Converts audio tensors to PCM int16 bytes.

    Applies a fade-in on the first chunk and crossfades at chunk boundaries
    to eliminate seam artifacts between streaming chunks.
    """

    def __init__(self, sample_rate: int = _DEFAULT_SR):
        self._sample_rate = sample_rate
        self._fade_in = _build_fade_in(sample_rate)
        self._crossfade_len = sample_rate * _CROSSFADE_MS // 1000  # 480 @ 24kHz
        self._prev_tail: Optional[torch.Tensor] = None

    def process(self, audio: torch.Tensor, chunk_index: int) -> bytes:
        # Ensure 1D float on CPU
        wav = audio.detach().cpu().squeeze().float()

        # Fade-in on first chunk only
        if chunk_index == 0:
            fade_len = min(len(self._fade_in), wav.size(0))
            wav[:fade_len] = wav[:fade_len] * self._fade_in[:fade_len]

        # Crossfade with previous chunk's tail
        if self._prev_tail is not None and wav.size(0) >= self._crossfade_len:
            xf_len = min(self._crossfade_len, self._prev_tail.size(0), wav.size(0))
            # Linear crossfade ramps
            fade_out = torch.linspace(1.0, 0.0, xf_len)
            fade_in = torch.linspace(0.0, 1.0, xf_len)
            # Blend the overlap zone
            blended = self._prev_tail[:xf_len] * fade_out + wav[:xf_len] * fade_in
            wav = torch.cat([blended, wav[xf_len:]])

        # Save tail for crossfade with next chunk (non-final only)
        if wav.size(0) > self._crossfade_len:
            self._prev_tail = wav[-self._crossfade_len:].clone()
            wav = wav[:-self._crossfade_len]  # trim tail (will be blended into next)
        else:
            self._prev_tail = None

        # Clamp and convert to int16
        wav = wav.clamp(-1.0, 1.0)
        pcm = (wav * 32767).to(torch.int16)
        return pcm.numpy().tobytes()

    def finalize(self) -> bytes:
        """Flush any buffered tail as the final PCM output."""
        if self._prev_tail is None:
            return b""
        wav = self._prev_tail.clamp(-1.0, 1.0)
        pcm = (wav * 32767).to(torch.int16)
        self._prev_tail = None
        return pcm.numpy().tobytes()

    def reset(self) -> None:
        """Clear crossfade state between generations."""
        self._prev_tail = None
