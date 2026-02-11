"""Data types for the streaming pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class ChunkStrategy(Enum):
    """Strategy for determining when to yield token chunks."""
    ADAPTIVE = "adaptive"
    SENTENCE_ALIGNED = "sentence_aligned"


@dataclass
class StreamConfig:
    """Configuration for the streaming pipeline."""
    first_chunk_tokens: int = 25
    subsequent_chunk_tokens: int = 75
    overlap_tokens: int = 3
    strategy: ChunkStrategy = ChunkStrategy.ADAPTIVE
    # T3 sampling parameters
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    max_gen_len: int = 1000


@dataclass
class TokenChunk:
    """A chunk of speech tokens from the T3 generator."""
    tokens: torch.Tensor  # shape (1, num_tokens)
    is_final: bool
    chunk_index: int


@dataclass
class AudioChunk:
    """A chunk of audio from the streaming pipeline."""
    pcm_bytes: bytes
    sample_rate: int
    is_final: bool
    chunk_index: int
    duration_ms: float
    audio_bytes: bytes = b""  # Encoded audio (e.g. OGG/Opus), empty if PCM-only
