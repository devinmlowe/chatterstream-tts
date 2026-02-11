"""Tests for ChunkVocoder — S3Gen per-chunk synthesis with causal state."""

import torch
import pytest
from unittest.mock import MagicMock, call

from chatterstream.types import TokenChunk


def _make_vocoder(s3gen=None):
    from chatterstream.chunk_vocoder import ChunkVocoder
    if s3gen is None:
        s3gen = _make_mock_s3gen()
    return ChunkVocoder(s3gen), s3gen


def _make_mock_s3gen():
    mock = MagicMock()
    mock.device = torch.device("cpu")
    mock.dtype = torch.float32
    mock.meanflow = True

    def flow_side_effect(speech_tokens, ref_dict=None, finalize=False, **kw):
        n_tokens = speech_tokens.size(-1)
        mel_frames = n_tokens * 2
        if not finalize:
            mel_frames -= 6
        return torch.randn(1, 80, max(mel_frames, 1))

    def hift_side_effect(speech_feat, cache_source=None):
        mel_frames = speech_feat.size(-1)
        wav = torch.randn(1, mel_frames * 120)
        source = torch.randn(1, 1, mel_frames * 10)
        return wav, source

    mock.flow_inference = MagicMock(side_effect=flow_side_effect)
    mock.hift_inference = MagicMock(side_effect=hift_side_effect)
    return mock


class TestChunkVocoderFinalize:
    """Verify finalize workaround: always True to flow, manual mel trim for non-final."""

    def test_always_passes_finalize_true_to_flow(self):
        vocoder, s3gen = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        vocoder.vocode(chunk, ref_dict={"dummy": True}, finalize=False)
        s3gen.flow_inference.assert_called_once()
        _, kwargs = s3gen.flow_inference.call_args
        # Always True to avoid upstream mask bug
        assert kwargs["finalize"] is True

    def test_final_chunk_passes_finalize_true(self):
        vocoder, s3gen = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=True, chunk_index=1)
        vocoder.vocode(chunk, ref_dict={"dummy": True}, finalize=True)
        _, kwargs = s3gen.flow_inference.call_args
        assert kwargs["finalize"] is True


class TestChunkVocoderCacheSource:
    """Verify cache_source chaining between chunks."""

    def test_first_chunk_cache_source_none(self):
        vocoder, s3gen = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        vocoder.vocode(chunk, ref_dict={}, finalize=False)
        _, kwargs = s3gen.hift_inference.call_args
        assert kwargs["cache_source"] is None

    def test_subsequent_chunk_gets_previous_source(self):
        vocoder, s3gen = _make_vocoder()
        chunk0 = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        chunk1 = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=1)
        vocoder.vocode(chunk0, ref_dict={}, finalize=False)
        vocoder.vocode(chunk1, ref_dict={}, finalize=False)
        # Second hift_inference call should have cache_source from first
        second_call_kwargs = s3gen.hift_inference.call_args_list[1][1]
        assert second_call_kwargs["cache_source"] is not None
        assert isinstance(second_call_kwargs["cache_source"], torch.Tensor)

    def test_cache_source_chaining_three_chunks(self):
        vocoder, s3gen = _make_vocoder()
        for i in range(3):
            chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=(i == 2), chunk_index=i)
            vocoder.vocode(chunk, ref_dict={}, finalize=(i == 2))
        # First: None, second and third: tensor
        calls = s3gen.hift_inference.call_args_list
        assert calls[0][1]["cache_source"] is None
        assert isinstance(calls[1][1]["cache_source"], torch.Tensor)
        assert isinstance(calls[2][1]["cache_source"], torch.Tensor)


class TestChunkVocoderReset:
    """Verify reset clears state."""

    def test_reset_clears_cache_source(self):
        vocoder, s3gen = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        vocoder.vocode(chunk, ref_dict={}, finalize=False)
        vocoder.reset()
        # Next vocode should start fresh
        chunk2 = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        vocoder.vocode(chunk2, ref_dict={}, finalize=False)
        second_call_kwargs = s3gen.hift_inference.call_args_list[-1][1]
        assert second_call_kwargs["cache_source"] is None


class TestChunkVocoderOutput:
    """Verify output tensor shapes."""

    def test_output_is_tensor(self):
        vocoder, _ = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        result = vocoder.vocode(chunk, ref_dict={}, finalize=False)
        assert isinstance(result, torch.Tensor)

    def test_output_is_1d_or_2d(self):
        vocoder, _ = _make_vocoder()
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        result = vocoder.vocode(chunk, ref_dict={}, finalize=False)
        assert result.dim() in (1, 2)


class TestChunkVocoderRefDict:
    """Verify ref_dict is passed through correctly."""

    def test_ref_dict_passed_to_flow(self):
        vocoder, s3gen = _make_vocoder()
        ref_dict = {"prompt_token": torch.randint(0, 100, (1, 50)), "embedding": torch.randn(1, 80)}
        chunk = TokenChunk(tokens=torch.randint(0, 100, (1, 25)), is_final=False, chunk_index=0)
        vocoder.vocode(chunk, ref_dict=ref_dict, finalize=False)
        _, kwargs = s3gen.flow_inference.call_args
        assert kwargs["ref_dict"] is ref_dict
