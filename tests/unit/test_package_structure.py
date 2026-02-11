"""Smoke tests: verify package imports and type construction."""

import torch
import pytest


class TestPackageImports:
    def test_import_package(self):
        import chatterstream

    def test_import_types(self):
        from chatterstream.types import (
            AudioChunk,
            ChunkStrategy,
            StreamConfig,
            TokenChunk,
        )

    def test_import_interfaces(self):
        from chatterstream.interfaces import (
            AudioPostProcessorBase,
            ChunkVocoderBase,
            ConditioningCacheBase,
            StreamingPipelineBase,
            TextProcessorBase,
            TokenGeneratorBase,
        )

    def test_import_streaming_tts(self):
        from chatterstream.streaming_tts import StreamingTTS

    def test_top_level_exports(self):
        from chatterstream import (
            StreamingTTS,
            AudioChunk,
            ChunkStrategy,
            StreamConfig,
            TokenChunk,
            AudioPostProcessorBase,
            ChunkVocoderBase,
            ConditioningCacheBase,
            StreamingPipelineBase,
            TextProcessorBase,
            TokenGeneratorBase,
        )

    def test_opus_encoder_not_at_top_level(self):
        """OpusEncoder requires av — not exported at top level."""
        import chatterstream
        assert not hasattr(chatterstream, "OpusEncoder")

    def test_hls_segmenter_not_at_top_level(self):
        """HLSSegmenter requires av — not exported at top level."""
        import chatterstream
        assert not hasattr(chatterstream, "HLSSegmenter")


class TestDataTypes:
    def test_stream_config_defaults(self):
        from chatterstream.types import StreamConfig, ChunkStrategy
        cfg = StreamConfig()
        assert cfg.first_chunk_tokens == 25
        assert cfg.subsequent_chunk_tokens == 75
        assert cfg.overlap_tokens == 3
        assert cfg.strategy == ChunkStrategy.ADAPTIVE
        assert cfg.temperature == 0.8

    def test_token_chunk(self):
        from chatterstream.types import TokenChunk
        tokens = torch.randint(0, 100, (1, 10))
        chunk = TokenChunk(tokens=tokens, is_final=False, chunk_index=0)
        assert chunk.tokens.shape == (1, 10)
        assert not chunk.is_final
        assert chunk.chunk_index == 0

    def test_audio_chunk(self):
        from chatterstream.types import AudioChunk
        chunk = AudioChunk(
            pcm_bytes=b"\x00" * 100,
            sample_rate=24000,
            is_final=True,
            chunk_index=3,
            duration_ms=50.0,
        )
        assert len(chunk.pcm_bytes) == 100
        assert chunk.sample_rate == 24000
        assert chunk.is_final
        assert chunk.chunk_index == 3

    def test_chunk_strategy_enum(self):
        from chatterstream.types import ChunkStrategy
        assert ChunkStrategy.ADAPTIVE.value == "adaptive"
        assert ChunkStrategy.SENTENCE_ALIGNED.value == "sentence_aligned"
