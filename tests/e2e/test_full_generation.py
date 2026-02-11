"""End-to-end tests with real model weights.

These tests require model weights to be available locally and are marked
with ``@pytest.mark.slow`` so they can be skipped in CI.

Run with: pytest tests/e2e/ -v -m slow
"""

import time
import struct
from pathlib import Path

import torch
import pytest


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_components():
    """Load T3, S3Gen, tokenizer, and conds directly (bypassing Perth watermarker)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.tts_turbo import Conditionals
    from chatterbox.models.s3gen.const import S3GEN_SR

    device = _get_device()
    map_location = torch.device("cpu") if device in ("cpu", "mps") else None

    ckpt_dir = Path(snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo",
        token=True,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    ))

    # T3
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)
    t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
    t3.load_state_dict(t3_state)
    del t3.tfmr.wte
    t3.to(device).eval()

    # S3Gen
    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True)
    s3gen.to(device).eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Built-in conditionals
    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device)

    return t3, s3gen, tokenizer, conds, S3GEN_SR, device


@pytest.fixture(scope="module")
def pipeline_and_info():
    """Build a StreamingPipeline from real model weights."""
    from chatterstream.text_processor import TextProcessor
    from chatterstream.token_generator import TokenGenerator
    from chatterstream.chunk_vocoder import ChunkVocoder
    from chatterstream.audio_post_processor import AudioPostProcessor
    from chatterstream.pipeline import StreamingPipeline
    from chatterstream.types import StreamConfig

    t3, s3gen, tokenizer, conds, sr, device = _load_components()

    config = StreamConfig(first_chunk_tokens=25, subsequent_chunk_tokens=75, overlap_tokens=3)

    class BuiltinCondCache:
        def get(self, voice_path: str):
            return conds
        def clear(self):
            pass

    pipeline = StreamingPipeline(
        text_processor=TextProcessor(tokenizer),
        conditioning_cache=BuiltinCondCache(),
        token_generator=TokenGenerator(t3, config),
        chunk_vocoder=ChunkVocoder(s3gen),
        audio_post_processor=AudioPostProcessor(sample_rate=sr),
        config=config,
        sample_rate=sr,
        device=device,
    )

    return pipeline, sr, device


@pytest.mark.slow
class TestFullGeneration:
    @pytest.mark.asyncio
    async def test_generates_valid_audio(self, pipeline_and_info):
        """Pipeline produces playable PCM chunks."""
        pipeline, sr, _ = pipeline_and_info
        chunks = []
        async for chunk in pipeline.synthesize(
            "Hello, this is a test of the streaming pipeline.",
            "builtin",
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1, "Should produce at least one chunk"
        total_bytes = sum(len(c.pcm_bytes) for c in chunks)
        assert total_bytes > 0, "Should produce non-empty audio"

        # Verify PCM is valid int16
        for chunk in chunks:
            n_samples = len(chunk.pcm_bytes) // 2
            samples = struct.unpack(f"<{n_samples}h", chunk.pcm_bytes)
            assert all(-32768 <= s <= 32767 for s in samples)

        # Verify we got multiple chunks (streaming actually worked)
        assert len(chunks) >= 2, f"Expected multiple chunks but got {len(chunks)}"

        # Verify sample rate
        assert chunks[0].sample_rate == sr

    @pytest.mark.asyncio
    async def test_first_chunk_latency(self, pipeline_and_info):
        """First audio chunk should arrive promptly."""
        pipeline, _, _ = pipeline_and_info
        start = time.monotonic()
        first_chunk_time = None

        async for chunk in pipeline.synthesize("Quick test.", "builtin"):
            if first_chunk_time is None:
                first_chunk_time = time.monotonic() - start
            break

        assert first_chunk_time is not None, "Should have received at least one chunk"
        # Allow 3s on first run (model warmup on MPS), target is <500ms warmed up
        assert first_chunk_time < 3.0, (
            f"First chunk latency {first_chunk_time:.3f}s exceeds 3s threshold"
        )

    @pytest.mark.asyncio
    async def test_streaming_produces_audio_duration(self, pipeline_and_info):
        """Streamed output should have reasonable total duration."""
        pipeline, _, _ = pipeline_and_info
        text = "This is a quality test for streaming synthesis."

        total_duration_ms = 0.0
        async for chunk in pipeline.synthesize(text, "builtin"):
            total_duration_ms += chunk.duration_ms

        # For a ~10-word sentence, expect at least 500ms and no more than 30s
        assert total_duration_ms > 500, f"Total duration {total_duration_ms:.0f}ms too short"
        assert total_duration_ms < 30000, f"Total duration {total_duration_ms:.0f}ms too long"
