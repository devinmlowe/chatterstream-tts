"""Factory function for creating a StreamingPipeline from a ChatterboxTurboTTS model."""

from __future__ import annotations

from typing import Optional

from .audio_post_processor import AudioPostProcessor
from .chunk_vocoder import ChunkVocoder
from .conditioning_cache import ConditioningCache
from .pipeline import StreamingPipeline
from .text_processor import TextProcessor
from .token_generator import TokenGenerator
from .types import StreamConfig


def create_streaming_pipeline(
    model,
    config: Optional[StreamConfig] = None,
) -> StreamingPipeline:
    """Create a StreamingPipeline from a ChatterboxTurboTTS model instance.

    Args:
        model: A ``ChatterboxTurboTTS`` instance (or compatible object with
            ``.t3``, ``.s3gen``, ``.tokenizer``, ``.device``, ``.sr`` attrs).
        config: Optional streaming configuration. Uses defaults if not provided.

    Returns:
        A fully wired ``StreamingPipeline`` ready for ``synthesize()`` calls.
    """
    config = config or StreamConfig()

    text_processor = TextProcessor(model.tokenizer)
    conditioning_cache = ConditioningCache(model)
    token_generator = TokenGenerator(model.t3, config)
    chunk_vocoder = ChunkVocoder(model.s3gen)
    audio_post_processor = AudioPostProcessor(sample_rate=model.sr)

    return StreamingPipeline(
        text_processor=text_processor,
        conditioning_cache=conditioning_cache,
        token_generator=token_generator,
        chunk_vocoder=chunk_vocoder,
        audio_post_processor=audio_post_processor,
        config=config,
        sample_rate=model.sr,
        device=model.device,
    )
