"""Streaming pipeline: async orchestrator wiring all components."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from .interfaces import (
    AudioPostProcessorBase,
    ChunkVocoderBase,
    ConditioningCacheBase,
    StreamingPipelineBase,
    TextProcessorBase,
    TokenGeneratorBase,
)
from .types import AudioChunk, StreamConfig


class StreamingPipeline(StreamingPipelineBase):
    """Async orchestrator that wires TextProcessor, ConditioningCache,
    TokenGenerator, ChunkVocoder, and AudioPostProcessor into a streaming
    pipeline yielding AudioChunks.
    """

    def __init__(
        self,
        text_processor: TextProcessorBase,
        conditioning_cache: ConditioningCacheBase,
        token_generator: TokenGeneratorBase,
        chunk_vocoder: ChunkVocoderBase,
        audio_post_processor: AudioPostProcessorBase,
        config: StreamConfig,
        sample_rate: int = 24000,
        device: str = "cpu",
    ):
        self._text_processor = text_processor
        self._cond_cache = conditioning_cache
        self._token_gen = token_generator
        self._vocoder = chunk_vocoder
        self._post_proc = audio_post_processor
        self._config = config
        self._sample_rate = sample_rate
        self._device = device
        self._opus_encoder = None  # Optional OpusEncoder

    async def synthesize(
        self, text: str, voice: str
    ) -> AsyncIterator[AudioChunk]:
        # Reset state between generations
        self._vocoder.reset()
        if hasattr(self._post_proc, 'reset'):
            self._post_proc.reset()
        if self._opus_encoder is not None:
            self._opus_encoder.reset()

        # Process text → token IDs (on correct device)
        text_tokens = self._text_processor.process(text, device=self._device)

        # Get cached conditioning
        conds = self._cond_cache.get(voice)

        # Generate token chunks and vocode each
        loop = asyncio.get_event_loop()
        last_chunk_index = 0
        for token_chunk in self._token_gen.generate(
            text_tokens, conds.t3, self._config
        ):
            # Run vocoding in executor to avoid blocking
            wav = await loop.run_in_executor(
                None,
                self._vocoder.vocode,
                token_chunk,
                conds.gen,
                token_chunk.is_final,
            )

            # Post-process to PCM bytes
            pcm_bytes = self._post_proc.process(wav, token_chunk.chunk_index)
            last_chunk_index = token_chunk.chunk_index

            # Encode to OGG/Opus if encoder is available
            audio_bytes = b""
            if self._opus_encoder is not None:
                audio_bytes = self._opus_encoder.encode(pcm_bytes)

            # Calculate duration from PCM byte length
            n_samples = len(pcm_bytes) // 2  # int16 = 2 bytes per sample
            duration_ms = (n_samples / self._sample_rate) * 1000

            yield AudioChunk(
                pcm_bytes=pcm_bytes,
                sample_rate=self._sample_rate,
                is_final=token_chunk.is_final,
                chunk_index=token_chunk.chunk_index,
                duration_ms=duration_ms,
                audio_bytes=audio_bytes,
            )

        # Flush crossfade tail from post-processor
        tail_pcm = self._post_proc.finalize()
        if tail_pcm:
            tail_audio_bytes = b""
            if self._opus_encoder is not None:
                tail_audio_bytes = self._opus_encoder.encode(tail_pcm)
                tail_audio_bytes += self._opus_encoder.finalize()

            n_samples = len(tail_pcm) // 2
            duration_ms = (n_samples / self._sample_rate) * 1000
            yield AudioChunk(
                pcm_bytes=tail_pcm,
                sample_rate=self._sample_rate,
                is_final=True,
                chunk_index=last_chunk_index + 1,
                duration_ms=duration_ms,
                audio_bytes=tail_audio_bytes,
            )
        elif self._opus_encoder is not None:
            # No tail PCM, but still need to finalize Opus stream
            final_ogg = self._opus_encoder.finalize()
            if final_ogg:
                yield AudioChunk(
                    pcm_bytes=b"",
                    sample_rate=self._sample_rate,
                    is_final=True,
                    chunk_index=last_chunk_index + 1,
                    duration_ms=0.0,
                    audio_bytes=final_ogg,
                )

    async def cancel(self) -> None:
        self._token_gen.cancel()
