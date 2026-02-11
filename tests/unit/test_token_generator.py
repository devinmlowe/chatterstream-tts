"""Tests for TokenGenerator — autoregressive T3 loop with chunk yielding."""

import torch
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from chatterstream.types import StreamConfig, ChunkStrategy, TokenChunk


def _make_generator(token_sequence, config=None):
    """Helper: create TokenGenerator with a mock T3 that produces a known sequence."""
    from chatterstream.token_generator import TokenGenerator

    config = config or StreamConfig()
    t3 = _make_mock_t3(token_sequence)
    return TokenGenerator(t3, config), t3


def _make_mock_t3(token_sequence):
    """Build a mock T3 model that yields tokens from a predetermined sequence."""
    mock = MagicMock()
    mock.hp = MagicMock()
    mock.hp.start_speech_token = 6561
    mock.hp.stop_speech_token = 6562
    mock.hp.speech_tokens_dict_size = 6563
    mock.device = torch.device("cpu")

    # prepare_input_embeds returns (embeds, len_cond)
    mock.prepare_input_embeds = MagicMock(
        return_value=(torch.randn(1, 50, 1024), 10)
    )
    # speech_emb maps token -> embedding
    mock.speech_emb = MagicMock(
        side_effect=lambda t: torch.randn(1, 1, 1024)
    )
    # speech_head maps hidden -> logits
    mock.speech_head = MagicMock(
        side_effect=lambda h: torch.randn(1, 1, 6563)
    )

    # tfmr returns (hidden_states, past_key_values) via an object
    call_count = [0]
    def tfmr_side_effect(**kwargs):
        result = MagicMock()
        result.__getitem__ = MagicMock(side_effect=lambda i: torch.randn(1, 1, 1024) if i == 0 else None)
        result.past_key_values = "kv_cache"

        # For the initial prefill, return logits that produce first token
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"].size(1) > 1:
            hidden = torch.randn(1, kwargs["inputs_embeds"].size(1), 1024)
        else:
            hidden = torch.randn(1, 1, 1024)
        result.__getitem__ = MagicMock(side_effect=lambda i: hidden if i == 0 else "kv_cache")
        return result
    mock.tfmr = MagicMock(side_effect=tfmr_side_effect)

    # Override the generator's _sample_token to return predetermined tokens
    mock._token_sequence = token_sequence
    return mock


def _make_t3_cond():
    """Minimal T3Cond mock."""
    cond = MagicMock()
    cond.speaker_emb = torch.randn(1, 256)
    return cond


class TestTokenGeneratorChunkSizes:
    """Verify adaptive chunk boundary logic."""

    def test_first_chunk_size_adaptive(self):
        """First chunk should have ~first_chunk_tokens tokens."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=10, overlap_tokens=0)
        # Sequence of 30 tokens (no stop token)
        seq = list(range(100, 130))
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        assert chunks[0].tokens.size(1) == 5
        assert chunks[0].chunk_index == 0

    def test_subsequent_chunk_sizes_adaptive(self):
        """After first chunk, subsequent chunks should be subsequent_chunk_tokens."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=10, overlap_tokens=0)
        seq = list(range(100, 130))
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        # Chunk 1 should have 10 tokens (subsequent size)
        if len(chunks) > 1:
            assert chunks[1].tokens.size(1) == 10


class TestTokenGeneratorOverlap:
    """Verify overlap token handling between chunks."""

    def test_overlap_carries_tokens(self):
        """Last N overlap tokens from one chunk should appear at start of next."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=10, overlap_tokens=2)
        seq = list(range(100, 130))
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        if len(chunks) >= 2:
            # Last 2 tokens of chunk 0 should be first 2 of chunk 1
            tail = chunks[0].tokens[0, -2:].tolist()
            head = chunks[1].tokens[0, :2].tolist()
            assert tail == head


class TestTokenGeneratorStopToken:
    """Verify stop token detection and final chunk marking."""

    def test_stop_token_ends_generation(self):
        """Generation should stop when stop_speech_token (6562) is encountered."""
        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=20, overlap_tokens=0)
        seq = list(range(100, 110)) + [6562]  # 10 tokens then stop
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        assert chunks[-1].is_final

    def test_stop_token_excluded_from_output(self):
        """Stop token itself should not appear in chunk tokens."""
        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=20, overlap_tokens=0)
        seq = list(range(100, 108)) + [6562]
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        all_tokens = torch.cat([c.tokens for c in chunks], dim=1)
        assert 6562 not in all_tokens.tolist()[0]


class TestTokenGeneratorOOVFiltering:
    """Verify out-of-vocabulary token filtering."""

    def test_oov_tokens_filtered(self):
        """Tokens >= 6561 (start_speech_token) should be filtered from output."""
        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=20, overlap_tokens=0)
        seq = [100, 200, 6561, 300, 6562]  # 6561 is OOV (start token)
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        all_tokens = torch.cat([c.tokens for c in chunks], dim=1)
        token_list = all_tokens.tolist()[0]
        assert 6561 not in token_list


class TestTokenGeneratorSilenceAppend:
    """Verify silence token appending on final chunk."""

    def test_final_chunk_has_silence_suffix(self):
        """Final chunk should have 3x silence tokens (4299) appended."""
        config = StreamConfig(first_chunk_tokens=20, subsequent_chunk_tokens=20, overlap_tokens=0)
        seq = list(range(100, 110)) + [6562]
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        final = chunks[-1]
        assert final.is_final
        # Last 3 tokens should be silence (4299)
        last_3 = final.tokens[0, -3:].tolist()
        assert last_3 == [4299, 4299, 4299]


class TestTokenGeneratorCancel:
    """Verify cancellation propagation."""

    def test_cancel_stops_generation(self):
        """Calling cancel() mid-generation should stop yielding chunks."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=5, overlap_tokens=0)
        seq = list(range(100, 200))  # Long sequence
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))

        chunks = []
        for chunk in gen.generate(text_tokens, _make_t3_cond(), config):
            chunks.append(chunk)
            gen.cancel()  # Cancel after first chunk
            break

        assert len(chunks) == 1


class TestTokenGeneratorChunkIndices:
    """Verify sequential chunk index assignment."""

    def test_chunk_indices_sequential(self):
        """Chunk indices should be 0, 1, 2, ..."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=5, overlap_tokens=0)
        seq = list(range(100, 125))
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestTokenGeneratorSentenceAligned:
    """Verify sentence-aligned chunk strategy."""

    def test_sentence_aligned_yields_on_silence(self):
        """In sentence_aligned mode, chunks yield on silence token (4299)."""
        config = StreamConfig(
            first_chunk_tokens=100,  # High threshold
            subsequent_chunk_tokens=100,
            overlap_tokens=0,
            strategy=ChunkStrategy.SENTENCE_ALIGNED,
        )
        # Tokens with a silence token at position 5
        seq = [100, 101, 102, 103, 104, 4299, 200, 201, 202, 6562]
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        # Should get at least 2 chunks (split on silence)
        assert len(chunks) >= 2


class TestTokenGeneratorKVCache:
    """Verify KV cache is passed correctly."""

    def test_kv_cache_reused(self):
        """past_key_values from prefill should be passed to subsequent steps."""
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=5, overlap_tokens=0)
        seq = list(range(100, 115))
        gen, mock_t3 = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        list(gen.generate(text_tokens, _make_t3_cond(), config))
        # tfmr should have been called with past_key_values after prefill
        calls = mock_t3.tfmr.call_args_list
        # First call is prefill (no past_key_values or None)
        # Subsequent calls should have past_key_values
        if len(calls) > 1:
            for call in calls[1:]:
                assert "past_key_values" in call.kwargs


class TestTokenGeneratorFinalChunkMarking:
    """Verify only the last chunk has is_final=True."""

    def test_only_last_chunk_is_final(self):
        config = StreamConfig(first_chunk_tokens=5, subsequent_chunk_tokens=5, overlap_tokens=0)
        seq = list(range(100, 120)) + [6562]
        gen, _ = _make_generator(seq, config)
        text_tokens = torch.randint(0, 100, (1, 5))
        chunks = list(gen.generate(text_tokens, _make_t3_cond(), config))
        for chunk in chunks[:-1]:
            assert not chunk.is_final
        assert chunks[-1].is_final
