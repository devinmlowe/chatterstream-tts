"""Chunk vocoder: per-chunk S3Gen synthesis with causal state management."""

from __future__ import annotations

from typing import Optional

import torch

from .interfaces import ChunkVocoderBase
from .types import TokenChunk

# Upstream flow.py constants for overlap trimming
_PRE_LOOKAHEAD_LEN = 3
_TOKEN_MEL_RATIO = 2
_MEL_OVERLAP_FRAMES = _PRE_LOOKAHEAD_LEN * _TOKEN_MEL_RATIO  # 6 mel frames

# HiFiGAN upsample ratio: mel frames → source signal samples
# S3Gen uses upsample_rates=[8,5,3] with istft hop_len=4 → 8*5*3*4 = 480
_HIFI_UPSAMPLE_RATIO = 480


class ChunkVocoder(ChunkVocoderBase):
    """Converts token chunks to audio waveforms using S3Gen.

    Manages ``cache_source`` state between chunks for phase-continuous
    HiFiGAN vocoding.

    Note: Always calls ``flow_inference`` with ``finalize=True`` and
    manually trims mel frames for non-final chunks.  The upstream
    ``finalize=False`` path has a mask/encoder shape mismatch; this
    workaround achieves the same overlap effect since the S3Gen
    decoder is fully causal (left-padding only).
    """

    def __init__(self, s3gen):
        self._s3gen = s3gen
        self._cache_source: Optional[torch.Tensor] = None

    def vocode(
        self,
        token_chunk: TokenChunk,
        ref_dict: dict,
        finalize: bool,
    ) -> torch.Tensor:
        # Token → Mel (always finalize=True to avoid upstream mask bug)
        mels = self._s3gen.flow_inference(
            speech_tokens=token_chunk.tokens,
            ref_dict=ref_dict,
            finalize=True,
        )

        # NOTE: We intentionally do NOT trim mel overlap frames here.
        # The upstream finalize=False path trims 6 mel frames (3 tokens *
        # 2 mel/token) from the end because the encoder lacks right-context
        # for those frames. However, trimming creates a hard mel boundary
        # between chunks that produces audible seam artifacts. Keeping the
        # full mel output (with slightly less contextual quality at the
        # boundary) sounds much smoother because cache_source maintains
        # waveform phase continuity and the causal decoder handles the rest.

        # Mel → Waveform via HiFiGAN with source continuity
        # HiFiGAN upsamples mel frames 480x to source signal (8*5*3*4 = 480).
        # If cache_source from a previous chunk is wider than the current chunk's
        # source buffer, upstream crashes. Truncate to fit.
        cache = self._cache_source
        if cache is not None:
            max_source_len = mels.size(-1) * _HIFI_UPSAMPLE_RATIO
            if cache.size(-1) > max_source_len:
                cache = cache[..., :max_source_len]
        wav, source = self._s3gen.hift_inference(
            speech_feat=mels,
            cache_source=cache,
        )
        self._cache_source = source

        return wav

    def reset(self) -> None:
        self._cache_source = None
