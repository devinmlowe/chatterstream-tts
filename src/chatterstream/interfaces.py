"""Abstract base classes defining contracts for each streaming component."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional

import torch

from .types import AudioChunk, StreamConfig, TokenChunk


class TextProcessorBase(ABC):
    """Normalizes text and tokenizes it into GPT2 token IDs."""

    @abstractmethod
    def process(self, text: str, device: torch.device | str = "cpu") -> torch.Tensor:
        """Normalize text and return token IDs tensor of shape (1, seq_len)."""
        ...


class ConditioningCacheBase(ABC):
    """Caches voice conditioning data keyed by file path."""

    @abstractmethod
    def get(self, voice_path: str) -> object:
        """Return Conditionals for the given voice file, using cache when possible."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear the entire cache."""
        ...


class TokenGeneratorBase(ABC):
    """Generates speech tokens autoregressively, yielding chunks mid-loop."""

    @abstractmethod
    def generate(
        self,
        text_tokens: torch.Tensor,
        t3_cond: object,
        config: StreamConfig,
    ) -> Iterator[TokenChunk]:
        """Yield TokenChunks as tokens are generated."""
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Signal the generator to stop at the next iteration."""
        ...


class ChunkVocoderBase(ABC):
    """Converts token chunks to audio tensors using S3Gen."""

    @abstractmethod
    def vocode(
        self,
        token_chunk: TokenChunk,
        ref_dict: dict,
        finalize: bool,
    ) -> torch.Tensor:
        """Convert a token chunk to an audio waveform tensor."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset vocoder state between full generations."""
        ...


class AudioPostProcessorBase(ABC):
    """Converts audio tensors to PCM int16 bytes with optional fade-in."""

    @abstractmethod
    def process(self, audio: torch.Tensor, chunk_index: int) -> bytes:
        """Convert audio tensor to PCM int16 bytes. Apply fade-in if chunk_index == 0."""
        ...

    @abstractmethod
    def finalize(self) -> bytes:
        """Flush any buffered audio (e.g. crossfade tail) as final PCM bytes."""
        ...


class StreamingPipelineBase(ABC):
    """Async orchestrator that wires all components together."""

    @abstractmethod
    async def synthesize(
        self, text: str, voice: str
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks for the given text and voice."""
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the current generation."""
        ...
