"""Integration tests: TokenGenerator → ChunkVocoder → AudioPostProcessor chain."""

import struct
from unittest.mock import MagicMock

import torch
import pytest

from chatterstream.types import StreamConfig, ChunkStrategy


def _make_mock_t3_for_integration(token_sequence):
    """Mock T3 that returns tokens from sequence via generate-compatible interface."""
    mock = MagicMock()
    mock.hp = MagicMock()
    mock.hp.start_speech_token = 6561
    mock.hp.stop_speech_token = 6562
    mock.hp.speech_tokens_dict_size = 6563
    mock.device = torch.device("cpu")

    mock.prepare_input_embeds = MagicMock(
        return_value=(torch.randn(1, 50, 1024), 10)
    )
    mock.speech_emb = MagicMock(
        side_effect=lambda t: torch.randn(1, 1, 1024)
    )

    # speech_head returns logits that deterministically produce tokens from sequence
    _idx = [0]
    def speech_head_fn(h):
        logits = torch.full((1, 1, 6563), -100.0)
        if _idx[0] < len(token_sequence):
            tok = token_sequence[_idx[0]]
            _idx[0] += 1
        else:
            tok = 6562  # stop
        logits[0, 0, tok] = 100.0  # Make this token overwhelmingly likely
        return logits
    mock.speech_head = MagicMock(side_effect=speech_head_fn)

    def tfmr_fn(**kwargs):
        result = MagicMock()
        embeds = kwargs.get("inputs_embeds")
        hidden = torch.randn(1, embeds.size(1) if embeds is not None else 1, 1024)
        result.__getitem__ = MagicMock(side_effect=lambda i: hidden if i == 0 else "kv")
        result.past_key_values = "kv"
        return result
    mock.tfmr = MagicMock(side_effect=tfmr_fn)

    return mock


def _make_mock_s3gen():
    """Mock S3Gen with deterministic outputs."""
    mock = MagicMock()
    mock.device = torch.device("cpu")
    mock.dtype = torch.float32
    mock.meanflow = True

    def flow_fn(speech_tokens, ref_dict=None, finalize=False, **kw):
        n = speech_tokens.size(-1) * 2
        if not finalize:
            n -= 6
        return torch.randn(1, 80, max(n, 1))

    def hift_fn(speech_feat, cache_source=None):
        wav = torch.randn(1, speech_feat.size(-1) * 120)
        source = torch.randn(1, 1, speech_feat.size(-1) * 10)
        return wav, source

    mock.flow_inference = MagicMock(side_effect=flow_fn)
    mock.hift_inference = MagicMock(side_effect=hift_fn)
    return mock


class TestTokenToAudioChain:
    """Test TokenGenerator → ChunkVocoder → AudioPostProcessor with mock models."""

    def test_chain_produces_bytes_for_each_chunk(self):
        from chatterstream.token_generator import TokenGenerator
        from chatterstream.chunk_vocoder import ChunkVocoder
        from chatterstream.audio_post_processor import AudioPostProcessor

        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=10, overlap_tokens=0)
        seq = list(range(100, 130)) + [6562]
        t3 = _make_mock_t3_for_integration(seq)
        s3gen = _make_mock_s3gen()

        token_gen = TokenGenerator(t3, config)
        vocoder = ChunkVocoder(s3gen)
        post_proc = AudioPostProcessor(sample_rate=24000)

        t3_cond = MagicMock()
        text_tokens = torch.randint(0, 100, (1, 5))

        audio_outputs = []
        for token_chunk in token_gen.generate(text_tokens, t3_cond, config):
            wav = vocoder.vocode(token_chunk, ref_dict={}, finalize=token_chunk.is_final)
            pcm = post_proc.process(wav, token_chunk.chunk_index)
            audio_outputs.append(pcm)

        assert len(audio_outputs) >= 2
        for pcm in audio_outputs:
            assert isinstance(pcm, bytes)
            assert len(pcm) > 0

    def test_chain_fade_in_only_first_chunk(self):
        from chatterstream.token_generator import TokenGenerator
        from chatterstream.chunk_vocoder import ChunkVocoder
        from chatterstream.audio_post_processor import AudioPostProcessor

        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=10, overlap_tokens=0)
        seq = list(range(100, 130)) + [6562]
        t3 = _make_mock_t3_for_integration(seq)
        s3gen = _make_mock_s3gen()

        # Use constant signal to verify fade
        def flow_const(speech_tokens, **kw):
            n = speech_tokens.size(-1) * 2
            if not kw.get("finalize", False):
                n -= 6
            return torch.ones(1, 80, max(n, 1))
        s3gen.flow_inference.side_effect = flow_const

        def hift_const(speech_feat, cache_source=None):
            wav = torch.ones(1, speech_feat.size(-1) * 120)
            source = torch.randn(1, 1, speech_feat.size(-1) * 10)
            return wav, source
        s3gen.hift_inference.side_effect = hift_const

        token_gen = TokenGenerator(t3, config)
        vocoder = ChunkVocoder(s3gen)
        post_proc = AudioPostProcessor(sample_rate=24000)

        t3_cond = MagicMock()
        text_tokens = torch.randint(0, 100, (1, 5))

        chunks_pcm = []
        for token_chunk in token_gen.generate(text_tokens, t3_cond, config):
            wav = vocoder.vocode(token_chunk, ref_dict={}, finalize=token_chunk.is_final)
            pcm = post_proc.process(wav, token_chunk.chunk_index)
            chunks_pcm.append(pcm)

        # First chunk should have faded start
        if len(chunks_pcm) >= 2:
            s0 = struct.unpack(f"<{len(chunks_pcm[0])//2}h", chunks_pcm[0])
            s1 = struct.unpack(f"<{len(chunks_pcm[1])//2}h", chunks_pcm[1])
            # First sample of chunk 0 should be near zero (faded)
            assert abs(s0[0]) < 100
            # First sample of chunk 1 should be large (not faded)
            assert abs(s1[0]) > 30000

    def test_vocoder_cache_source_chained(self):
        from chatterstream.token_generator import TokenGenerator
        from chatterstream.chunk_vocoder import ChunkVocoder
        from chatterstream.audio_post_processor import AudioPostProcessor

        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=10, overlap_tokens=0)
        seq = list(range(100, 130)) + [6562]
        t3 = _make_mock_t3_for_integration(seq)
        s3gen = _make_mock_s3gen()

        token_gen = TokenGenerator(t3, config)
        vocoder = ChunkVocoder(s3gen)
        post_proc = AudioPostProcessor(sample_rate=24000)

        t3_cond = MagicMock()
        text_tokens = torch.randint(0, 100, (1, 5))

        for token_chunk in token_gen.generate(text_tokens, t3_cond, config):
            vocoder.vocode(token_chunk, ref_dict={}, finalize=token_chunk.is_final)

        calls = s3gen.hift_inference.call_args_list
        # First call: cache_source=None
        assert calls[0][1]["cache_source"] is None
        # Subsequent calls: cache_source is a tensor
        for c in calls[1:]:
            assert isinstance(c[1]["cache_source"], torch.Tensor)

    def test_finalize_flag_correct(self):
        from chatterstream.token_generator import TokenGenerator
        from chatterstream.chunk_vocoder import ChunkVocoder

        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=10, overlap_tokens=0)
        seq = list(range(100, 130)) + [6562]
        t3 = _make_mock_t3_for_integration(seq)
        s3gen = _make_mock_s3gen()

        token_gen = TokenGenerator(t3, config)
        vocoder = ChunkVocoder(s3gen)

        t3_cond = MagicMock()
        text_tokens = torch.randint(0, 100, (1, 5))

        for token_chunk in token_gen.generate(text_tokens, t3_cond, config):
            vocoder.vocode(token_chunk, ref_dict={}, finalize=token_chunk.is_final)

        calls = s3gen.flow_inference.call_args_list
        # ChunkVocoder always passes finalize=True to flow (upstream bug workaround)
        for c in calls:
            assert c[1]["finalize"] is True
