"""Integration tests for OGG/Opus streaming through the full pipeline."""

import asyncio
import io

import av
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from chatterstream.types import AudioChunk, StreamConfig, TokenChunk
from chatterstream.opus_encoder import OpusEncoder


def _make_pipeline_with_opus(num_chunks=3):
    """Build a StreamingPipeline with mocked components and an OpusEncoder."""
    from chatterstream.pipeline import StreamingPipeline

    config = StreamConfig(
        first_chunk_tokens=5,
        subsequent_chunk_tokens=10,
        overlap_tokens=0,
    )

    text_processor = MagicMock()
    text_processor.process = MagicMock(
        return_value=torch.randint(0, 100, (1, 10))
    )

    cond_cache = MagicMock()
    mock_conds = MagicMock()
    mock_conds.t3 = MagicMock()
    mock_conds.gen = {"prompt_token": torch.randint(0, 100, (1, 50))}
    cond_cache.get = MagicMock(return_value=mock_conds)

    # Token generator yields chunks
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

    # Vocoder returns audio-like tensor (1 second of sine at 440Hz per chunk)
    def _vocode(chunk, ref_dict, finalize):
        n = 24000  # 1 second at 24kHz
        t = np.linspace(0, 2 * np.pi * 440, n)
        return torch.tensor(np.sin(t) * 0.5, dtype=torch.float32).unsqueeze(0)

    vocoder = MagicMock()
    vocoder.vocode = MagicMock(side_effect=_vocode)
    vocoder.reset = MagicMock()

    # Post-processor converts to PCM int16 bytes
    def _process(audio, chunk_index):
        wav = audio.squeeze().clamp(-1.0, 1.0)
        pcm = (wav * 32767).to(torch.int16)
        return pcm.numpy().tobytes()

    post_proc = MagicMock()
    post_proc.process = MagicMock(side_effect=_process)
    post_proc.finalize = MagicMock(return_value=b"")
    post_proc.reset = MagicMock()

    pipeline = StreamingPipeline(
        text_processor=text_processor,
        conditioning_cache=cond_cache,
        token_generator=token_gen,
        chunk_vocoder=vocoder,
        audio_post_processor=post_proc,
        config=config,
        sample_rate=24000,
    )
    # Wire Opus encoder
    pipeline._opus_encoder = OpusEncoder(input_sample_rate=24000, bitrate=64000)

    return pipeline


class TestPipelineProducesValidOgg:
    @pytest.mark.asyncio
    async def test_full_stream_is_parseable_ogg(self):
        """Full pipeline with OpusEncoder produces a valid OGG/Opus stream."""
        pipeline = _make_pipeline_with_opus(num_chunks=3)
        ogg_parts = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg_parts.append(chunk.audio_bytes)

        ogg_data = b"".join(ogg_parts)
        assert len(ogg_data) > 0

        # Parse as OGG/Opus
        container = av.open(io.BytesIO(ogg_data), mode="r")
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        container.close()

    @pytest.mark.asyncio
    async def test_ogg_starts_with_sync_pattern(self):
        """OGG stream starts with the OggS sync pattern."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)
        first_bytes = b""

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                first_bytes = chunk.audio_bytes
                break

        assert first_bytes[:4] == b"OggS"


class TestOpusStreamPlayable:
    @pytest.mark.asyncio
    async def test_decode_back_to_pcm(self):
        """OGG stream can be decoded back to audio samples."""
        pipeline = _make_pipeline_with_opus(num_chunks=3)
        ogg_parts = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg_parts.append(chunk.audio_bytes)

        ogg_data = b"".join(ogg_parts)
        container = av.open(io.BytesIO(ogg_data), mode="r")

        total_samples = 0
        for frame in container.decode(audio=0):
            total_samples += frame.samples

        container.close()

        # 3 chunks of ~1 second each at 48kHz (resampled from 24kHz)
        # Opus may add/remove samples due to codec delay, so check roughly
        assert total_samples > 48000  # At least 1 second
        assert total_samples < 48000 * 5  # Not more than 5 seconds

    @pytest.mark.asyncio
    async def test_output_is_48khz(self):
        """Opus output is always 48kHz."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)
        ogg_parts = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg_parts.append(chunk.audio_bytes)

        ogg_data = b"".join(ogg_parts)
        container = av.open(io.BytesIO(ogg_data), mode="r")
        assert container.streams.audio[0].rate == 48000
        container.close()

    @pytest.mark.asyncio
    async def test_output_is_mono(self):
        """Opus output is mono."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)
        ogg_parts = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg_parts.append(chunk.audio_bytes)

        ogg_data = b"".join(ogg_parts)
        container = av.open(io.BytesIO(ogg_data), mode="r")
        assert container.streams.audio[0].channels == 1
        container.close()


class TestMultipleGenerationsReset:
    @pytest.mark.asyncio
    async def test_two_generations_produce_independent_streams(self):
        """Two synthesis runs produce independently valid OGG streams."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)

        # First generation
        ogg1_parts = []
        async for chunk in pipeline.synthesize("First.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg1_parts.append(chunk.audio_bytes)
        ogg1 = b"".join(ogg1_parts)

        # Reset token generator mock for second run
        chunks = [
            TokenChunk(tokens=torch.randint(0, 6000, (1, 10)), is_final=False, chunk_index=0),
            TokenChunk(tokens=torch.randint(0, 6000, (1, 10)), is_final=True, chunk_index=1),
        ]
        pipeline._token_gen.generate = MagicMock(return_value=iter(chunks))

        # Second generation (pipeline.synthesize resets Opus encoder internally)
        ogg2_parts = []
        async for chunk in pipeline.synthesize("Second.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg2_parts.append(chunk.audio_bytes)
        ogg2 = b"".join(ogg2_parts)

        # Both should be independently valid
        for ogg_data in (ogg1, ogg2):
            container = av.open(io.BytesIO(ogg_data), mode="r")
            assert container.streams.audio[0].codec_context.name == "opus"
            container.close()


class TestFinalizeProducesCompleteStream:
    @pytest.mark.asyncio
    async def test_stream_has_proper_ogg_pages(self):
        """OGG stream has header page + data pages + proper termination."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)
        ogg_parts = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            if chunk.audio_bytes:
                ogg_parts.append(chunk.audio_bytes)

        ogg_data = b"".join(ogg_parts)

        # Count OGG page sync patterns (OggS)
        page_count = 0
        pos = 0
        while True:
            idx = ogg_data.find(b"OggS", pos)
            if idx == -1:
                break
            page_count += 1
            pos = idx + 4

        # Minimum: 1 header page + 1 comment page + at least 1 data page
        assert page_count >= 3, f"Expected >= 3 OGG pages, got {page_count}"

    @pytest.mark.asyncio
    async def test_pcm_bytes_still_available(self):
        """PCM bytes are still produced alongside Opus encoding."""
        pipeline = _make_pipeline_with_opus(num_chunks=2)
        total_pcm = 0

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            total_pcm += len(chunk.pcm_bytes)

        # 2 chunks of ~24000 samples * 2 bytes/sample = ~96000 bytes
        assert total_pcm > 0

    @pytest.mark.asyncio
    async def test_last_chunk_is_final(self):
        """The last AudioChunk yielded has is_final=True."""
        pipeline = _make_pipeline_with_opus(num_chunks=3)
        chunks = []

        async for chunk in pipeline.synthesize("Test.", "/path/to/voice.wav"):
            chunks.append(chunk)

        assert chunks[-1].is_final
