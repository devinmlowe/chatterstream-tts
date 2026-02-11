"""Tests for StreamingPipeline — async orchestrator wiring all components."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import torch
import pytest

from chatterstream.types import AudioChunk, StreamConfig, TokenChunk


def _make_pipeline(
    num_chunks=3,
    token_seq=None,
    config=None,
):
    """Build a StreamingPipeline with fully mocked components."""
    from chatterstream.pipeline import StreamingPipeline

    config = config or StreamConfig(
        first_chunk_tokens=5,
        subsequent_chunk_tokens=10,
        overlap_tokens=0,
    )

    # Mock text processor
    text_processor = MagicMock()
    text_processor.process = MagicMock(
        return_value=torch.randint(0, 100, (1, 10))
    )

    # Mock conditioning cache
    cond_cache = MagicMock()
    mock_conds = MagicMock()
    mock_conds.t3 = MagicMock()
    mock_conds.gen = {"prompt_token": torch.randint(0, 100, (1, 50))}
    cond_cache.get = MagicMock(return_value=mock_conds)

    # Mock token generator — yields predetermined chunks
    token_gen = MagicMock()
    chunks = []
    for i in range(num_chunks):
        chunks.append(TokenChunk(
            tokens=torch.randint(0, 6000, (1, 10)),
            is_final=(i == num_chunks - 1),
            chunk_index=i,
        ))
    token_gen.generate = MagicMock(return_value=iter(chunks))
    token_gen.cancel = MagicMock()

    # Mock chunk vocoder
    vocoder = MagicMock()
    vocoder.vocode = MagicMock(
        side_effect=lambda chunk, ref_dict, finalize: torch.randn(1, 2400)
    )
    vocoder.reset = MagicMock()

    # Mock post-processor
    post_proc = MagicMock()
    post_proc.process = MagicMock(
        side_effect=lambda audio, chunk_index: b"\x00" * 4800
    )
    post_proc.finalize = MagicMock(return_value=b"")

    pipeline = StreamingPipeline(
        text_processor=text_processor,
        conditioning_cache=cond_cache,
        token_generator=token_gen,
        chunk_vocoder=vocoder,
        audio_post_processor=post_proc,
        config=config,
        sample_rate=24000,
    )

    return pipeline, {
        "text_processor": text_processor,
        "cond_cache": cond_cache,
        "token_gen": token_gen,
        "vocoder": vocoder,
        "post_proc": post_proc,
    }


class TestStreamingPipelineChunkYielding:
    @pytest.mark.asyncio
    async def test_yields_correct_number_of_chunks(self):
        pipeline, _ = _make_pipeline(num_chunks=3)
        chunks = []
        async for chunk in pipeline.synthesize("Hello world.", "/path/to/voice.wav"):
            chunks.append(chunk)
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_chunks_are_audio_chunks(self):
        pipeline, _ = _make_pipeline(num_chunks=2)
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            assert isinstance(chunk, AudioChunk)


class TestStreamingPipelineFinalChunk:
    @pytest.mark.asyncio
    async def test_last_chunk_is_final(self):
        pipeline, _ = _make_pipeline(num_chunks=3)
        chunks = []
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            chunks.append(chunk)
        assert chunks[-1].is_final
        for c in chunks[:-1]:
            assert not c.is_final


class TestStreamingPipelineCancel:
    @pytest.mark.asyncio
    async def test_cancel_propagates_to_token_generator(self):
        pipeline, mocks = _make_pipeline(num_chunks=5)
        # Start synthesis, cancel after first chunk
        count = 0
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            count += 1
            if count == 1:
                await pipeline.cancel()
                break
        mocks["token_gen"].cancel.assert_called_once()


class TestStreamingPipelineVocoderReset:
    @pytest.mark.asyncio
    async def test_vocoder_reset_at_start(self):
        pipeline, mocks = _make_pipeline(num_chunks=2)
        async for _ in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            pass
        mocks["vocoder"].reset.assert_called_once()


class TestStreamingPipelineSampleRate:
    @pytest.mark.asyncio
    async def test_chunk_sample_rate(self):
        pipeline, _ = _make_pipeline(num_chunks=1)
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            assert chunk.sample_rate == 24000


class TestStreamingPipelineDuration:
    @pytest.mark.asyncio
    async def test_chunk_duration_calculated(self):
        pipeline, _ = _make_pipeline(num_chunks=1)
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            # 4800 bytes / 2 (int16) = 2400 samples, at 24kHz = 100ms
            assert chunk.duration_ms == pytest.approx(100.0)


class TestStreamingPipelineChunkIndices:
    @pytest.mark.asyncio
    async def test_sequential_chunk_indices(self):
        pipeline, _ = _make_pipeline(num_chunks=4)
        indices = []
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            indices.append(chunk.chunk_index)
        assert indices == [0, 1, 2, 3]


class TestStreamingPipelineEmptyText:
    @pytest.mark.asyncio
    async def test_empty_text_still_produces_output(self):
        """Empty text gets default text from TextProcessor."""
        pipeline, _ = _make_pipeline(num_chunks=1)
        chunks = []
        async for chunk in pipeline.synthesize("", "/path/to/voice.wav"):
            chunks.append(chunk)
        # TextProcessor handles empty → default, so pipeline still works
        assert len(chunks) == 1


class TestStreamingPipelineCacheUsage:
    @pytest.mark.asyncio
    async def test_conditioning_cache_called(self):
        pipeline, mocks = _make_pipeline(num_chunks=1)
        async for _ in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            pass
        mocks["cond_cache"].get.assert_called_once_with("/path/to/voice.wav")


class TestStreamingPipelineFinalize:
    @pytest.mark.asyncio
    async def test_post_proc_finalize_called(self):
        """Pipeline should call finalize() on post-processor after last chunk."""
        pipeline, mocks = _make_pipeline(num_chunks=2)
        mocks["post_proc"].finalize = MagicMock(return_value=b"")
        async for _ in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            pass
        mocks["post_proc"].finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_tail_yielded_when_nonempty(self):
        """If finalize returns PCM bytes, a trailing AudioChunk is yielded."""
        pipeline, mocks = _make_pipeline(num_chunks=1)
        mocks["post_proc"].finalize = MagicMock(return_value=b"\x00" * 960)
        chunks = []
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            chunks.append(chunk)
        # 1 chunk from generator + 1 tail chunk from finalize
        assert len(chunks) == 2
        assert chunks[-1].is_final
        assert len(chunks[-1].pcm_bytes) == 960

    @pytest.mark.asyncio
    async def test_finalize_empty_tail_not_yielded(self):
        """If finalize returns empty bytes, no extra chunk is yielded."""
        pipeline, mocks = _make_pipeline(num_chunks=2)
        mocks["post_proc"].finalize = MagicMock(return_value=b"")
        chunks = []
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            chunks.append(chunk)
        assert len(chunks) == 2


class TestStreamingPipelineOpusEncoding:
    @pytest.mark.asyncio
    async def test_opus_encoding_when_encoder_provided(self):
        """AudioChunk.audio_bytes contains OGG data when encoder is set."""
        from chatterstream.opus_encoder import OpusEncoder
        pipeline, mocks = _make_pipeline(num_chunks=2)
        mocks["post_proc"].finalize = MagicMock(return_value=b"")
        pipeline._opus_encoder = OpusEncoder(input_sample_rate=24000)
        chunks = []
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            chunks.append(chunk)
        # At least one chunk should have audio_bytes
        has_ogg = any(chunk.audio_bytes for chunk in chunks)
        assert has_ogg

    @pytest.mark.asyncio
    async def test_no_opus_by_default(self):
        """Without encoder, audio_bytes should be empty."""
        pipeline, mocks = _make_pipeline(num_chunks=1)
        mocks["post_proc"].finalize = MagicMock(return_value=b"")
        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            assert chunk.audio_bytes == b""
