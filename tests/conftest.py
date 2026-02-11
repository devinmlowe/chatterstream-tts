"""Shared test fixtures for chatterstream tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock T3 model
# ---------------------------------------------------------------------------

@dataclass
class MockT3Cond:
    """Minimal T3Cond stand-in."""
    speaker_emb: torch.Tensor
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None
    cond_prompt_speech_emb: Optional[torch.Tensor] = None
    emotion_adv: Optional[torch.Tensor] = None

    def to(self, device=None):
        return self


class MockT3Config:
    """Minimal T3Config (hp) stand-in."""
    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 6563


class MockT3:
    """Mock T3 model for unit tests.

    Produces a configurable sequence of tokens then stops.
    """

    def __init__(self, token_sequence: list[int] | None = None):
        self.hp = MockT3Config()
        seq = token_sequence or list(range(100, 200))
        self._tokens = seq
        self.device = torch.device("cpu")

        # Expose sub-modules as mocks so attribute access works
        self.tfmr = MagicMock()
        self.speech_emb = MagicMock(
            side_effect=lambda t: torch.randn(1, 1, 1024)
        )
        self.speech_head = MagicMock(
            side_effect=lambda h: torch.randn(1, 1, self.hp.speech_tokens_dict_size)
        )
        self.text_emb = MagicMock(
            side_effect=lambda t: torch.randn(1, t.size(1), 1024)
        )

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
        seq_len = text_tokens.size(1) + speech_tokens.size(1) + 10  # cond + text + speech
        return torch.randn(1, seq_len, 1024), 10


# ---------------------------------------------------------------------------
# Mock S3Gen
# ---------------------------------------------------------------------------

class MockS3Gen:
    """Mock S3Gen model for unit tests."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.meanflow = True
        self._call_count = 0

    def flow_inference(
        self,
        speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=None,
        n_cfm_timesteps=None,
        finalize=False,
        speech_token_lens=None,
    ):
        num_tokens = speech_tokens.size(-1)
        mel_frames = num_tokens * 2
        if not finalize:
            mel_frames -= 6  # pre_lookahead_len * token_mel_ratio
        return torch.randn(1, 80, max(mel_frames, 1))

    def hift_inference(self, speech_feat, cache_source=None):
        mel_frames = speech_feat.size(-1)
        # 120x upsampling (8 * 5 * 3)
        num_samples = mel_frames * 120
        wav = torch.randn(1, num_samples)
        source = torch.randn(1, 1, mel_frames * 10)
        return wav, source


# ---------------------------------------------------------------------------
# Mock Conditionals
# ---------------------------------------------------------------------------

@dataclass
class MockConditionals:
    """Minimal Conditionals stand-in."""
    t3: MockT3Cond
    gen: dict

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Mock GPT2 tokenizer that returns predictable IDs."""

    def __init__(self):
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"

    def __call__(self, text, return_tensors=None, **kwargs):
        # Return 1 token per character (deterministic for tests)
        ids = [ord(c) % 256 for c in text]
        result = MagicMock()
        result.input_ids = torch.tensor([ids], dtype=torch.long)
        return result

    def __len__(self):
        return 50276


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_t3():
    return MockT3()


@pytest.fixture
def mock_s3gen():
    return MockS3Gen()


@pytest.fixture
def mock_t3_cond():
    return MockT3Cond(speaker_emb=torch.randn(1, 256))


@pytest.fixture
def mock_conditionals():
    return MockConditionals(
        t3=MockT3Cond(speaker_emb=torch.randn(1, 256)),
        gen={
            "prompt_token": torch.randint(0, 6561, (1, 50)),
            "prompt_token_len": torch.tensor([50]),
            "prompt_feat": torch.randn(1, 100, 80),
            "prompt_feat_len": torch.tensor([100]),
            "embedding": torch.randn(1, 80),
        },
    )


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def stream_config():
    from chatterstream.types import StreamConfig
    return StreamConfig()
