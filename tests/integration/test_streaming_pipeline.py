"""Integration tests: full StreamingPipeline with mock models."""

import asyncio
from unittest.mock import MagicMock

import torch
import pytest

from chatterstream.types import AudioChunk, StreamConfig, TokenChunk


def _build_integrated_pipeline(num_tokens=30):
    """Build pipeline with real component wiring but mock models."""
    from chatterstream.text_processor import TextProcessor
    from chatterstream.conditioning_cache import ConditioningCache
    from chatterstream.token_generator import TokenGenerator
    from chatterstream.chunk_vocoder import ChunkVocoder
    from chatterstream.audio_post_processor import AudioPostProcessor
    from chatterstream.pipeline import StreamingPipeline

    config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=10, overlap_tokens=0)

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.pad_token = "<|endoftext|>"

    def tokenize(text, **kw):
        r = MagicMock()
        r.input_ids = torch.randint(0, 100, (1, 10))
        return r
    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize

    text_processor = TextProcessor(tokenizer)

    # Mock model for conditioning cache
    mock_model = MagicMock()
    t3_cond = MagicMock()
    t3_cond.speaker_emb = torch.randn(1, 256)
    gen_dict = {"prompt_token": torch.randint(0, 100, (1, 50))}

    conds = MagicMock()
    conds.t3 = t3_cond
    conds.gen = gen_dict
    mock_model.prepare_conditionals = MagicMock(return_value=conds)
    cond_cache = ConditioningCache(mock_model)

    # Mock T3 for token generator
    token_seq = list(range(100, 100 + num_tokens)) + [6562]
    t3 = MagicMock()
    t3.hp = MagicMock()
    t3.hp.start_speech_token = 6561
    t3.hp.stop_speech_token = 6562
    t3.hp.speech_tokens_dict_size = 6563
    t3.device = torch.device("cpu")
    t3.prepare_input_embeds = MagicMock(
        return_value=(torch.randn(1, 50, 1024), 10)
    )
    t3.speech_emb = MagicMock(side_effect=lambda t: torch.randn(1, 1, 1024))

    _idx = [0]
    def speech_head_fn(h):
        logits = torch.full((1, 1, 6563), -100.0)
        if _idx[0] < len(token_seq):
            logits[0, 0, token_seq[_idx[0]]] = 100.0
            _idx[0] += 1
        else:
            logits[0, 0, 6562] = 100.0
        return logits
    t3.speech_head = MagicMock(side_effect=speech_head_fn)

    def tfmr_fn(**kwargs):
        result = MagicMock()
        embeds = kwargs.get("inputs_embeds")
        hidden = torch.randn(1, embeds.size(1) if embeds is not None else 1, 1024)
        result.__getitem__ = MagicMock(side_effect=lambda i: hidden if i == 0 else "kv")
        result.past_key_values = "kv"
        return result
    t3.tfmr = MagicMock(side_effect=tfmr_fn)

    token_gen = TokenGenerator(t3, config)

    # Mock S3Gen for vocoder
    s3gen = MagicMock()
    s3gen.device = torch.device("cpu")
    s3gen.flow_inference = MagicMock(
        side_effect=lambda speech_tokens, **kw: torch.randn(1, 80, speech_tokens.size(-1) * 2)
    )
    s3gen.hift_inference = MagicMock(
        side_effect=lambda speech_feat, **kw: (
            torch.randn(1, speech_feat.size(-1) * 120),
            torch.randn(1, 1, speech_feat.size(-1) * 10),
        )
    )
    vocoder = ChunkVocoder(s3gen)
    post_proc = AudioPostProcessor(sample_rate=24000)

    pipeline = StreamingPipeline(
        text_processor=text_processor,
        conditioning_cache=cond_cache,
        token_generator=token_gen,
        chunk_vocoder=vocoder,
        audio_post_processor=post_proc,
        config=config,
        sample_rate=24000,
    )

    return pipeline


class TestIntegratedPipeline:
    @pytest.mark.asyncio
    async def test_produces_audio_chunks(self, tmp_path):
        voice = tmp_path / "voice.wav"
        voice.write_bytes(b"fake")
        pipeline = _build_integrated_pipeline(num_tokens=25)
        chunks = []
        async for chunk in pipeline.synthesize("Hello world.", str(voice)):
            chunks.append(chunk)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, AudioChunk)
            assert len(c.pcm_bytes) > 0

    @pytest.mark.asyncio
    async def test_final_chunk_marked(self, tmp_path):
        voice = tmp_path / "voice.wav"
        voice.write_bytes(b"fake")
        pipeline = _build_integrated_pipeline(num_tokens=25)
        chunks = []
        async for chunk in pipeline.synthesize("Hello.", str(voice)):
            chunks.append(chunk)
        assert chunks[-1].is_final

    @pytest.mark.asyncio
    async def test_chunk_indices_sequential(self, tmp_path):
        voice = tmp_path / "voice.wav"
        voice.write_bytes(b"fake")
        pipeline = _build_integrated_pipeline(num_tokens=25)
        indices = []
        async for chunk in pipeline.synthesize("Hello.", str(voice)):
            indices.append(chunk.chunk_index)
        assert indices == list(range(len(indices)))

    @pytest.mark.asyncio
    async def test_sample_rate_consistent(self, tmp_path):
        voice = tmp_path / "voice.wav"
        voice.write_bytes(b"fake")
        pipeline = _build_integrated_pipeline(num_tokens=15)
        async for chunk in pipeline.synthesize("Test.", str(voice)):
            assert chunk.sample_rate == 24000
