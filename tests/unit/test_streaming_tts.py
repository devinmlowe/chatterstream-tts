"""Tests for StreamingTTS facade — the primary public API."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import torch

from chatterstream.streaming_tts import StreamingTTS
from chatterstream.types import AudioChunk, StreamConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(device="cpu"):
    """Create a mock ChatterboxTurboTTS-like object."""
    model = MagicMock()
    model.device = device
    model.sr = 24000
    model.t3 = MagicMock()
    model.t3.half = MagicMock(return_value=model.t3)
    model.s3gen = MagicMock()
    model.tokenizer = MagicMock()
    model.tokenizer.pad_token = "<|endoftext|>"
    model.tokenizer.eos_token = "<|endoftext|>"
    model.conds = MagicMock()
    model.conds.t3 = MagicMock()
    model.conds.t3.speaker_emb = torch.randn(1, 256)
    model.conds.t3.emotion_adv = None
    model.conds.gen = {"prompt_token": torch.randint(0, 100, (1, 10))}
    return model


def _patch_from_pretrained(model):
    """Patch ChatterboxTurboTTS.from_pretrained on the real upstream class.

    The module-level ``ChatterboxTurboTTS`` in streaming_tts.py starts as
    ``None`` (populated lazily by ``_ensure_upstream``), so we eagerly
    populate it then patch the classmethod on the resolved class.
    """
    from chatterstream.streaming_tts import _ensure_upstream
    _ensure_upstream()
    return patch(
        "chatterstream.streaming_tts.ChatterboxTurboTTS.from_pretrained",
        return_value=model,
    )


def _make_audio_chunk(index=0, is_final=False):
    """Create a test AudioChunk."""
    return AudioChunk(
        pcm_bytes=b"\x00\x01" * 50,
        sample_rate=24000,
        is_final=is_final,
        chunk_index=index,
        duration_ms=2.08,
    )


async def _mock_pipeline_synthesize(text, voice):
    """Mock async generator yielding test AudioChunks."""
    yield _make_audio_chunk(0, is_final=False)
    yield _make_audio_chunk(1, is_final=True)


# ---------------------------------------------------------------------------
# 1. Constructor stores config without loading
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_constructor_stores_config_without_loading(self):
        """StreamingTTS() stores settings without loading model."""
        tts = StreamingTTS(device="cpu", watermark=False)
        assert tts._device_request == "cpu"
        assert tts._watermark is False
        assert tts._model is None
        assert tts._pipeline is None

    def test_default_config_created(self):
        tts = StreamingTTS()
        assert isinstance(tts._config, StreamConfig)

    def test_custom_config_stored(self):
        cfg = StreamConfig(first_chunk_tokens=10)
        tts = StreamingTTS(config=cfg)
        assert tts._config.first_chunk_tokens == 10


# ---------------------------------------------------------------------------
# 2. is_loaded false before load
# ---------------------------------------------------------------------------

class TestIsLoaded:
    def test_is_loaded_false_before_load(self):
        tts = StreamingTTS()
        assert tts.is_loaded is False

    def test_is_loaded_true_after_load(self):
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)
        with _patch_from_pretrained(model):
            tts.load()
        assert tts.is_loaded is True


# ---------------------------------------------------------------------------
# 3 & 4. load() — returns self, calls from_pretrained
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_self_for_chaining(self):
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)
        with _patch_from_pretrained(model):
            result = tts.load()
        assert result is tts

    def test_load_calls_from_pretrained(self):
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)
        with _patch_from_pretrained(model) as mock_fp:
            tts.load()
            mock_fp.assert_called_once_with("cpu")

    def test_builds_pipeline(self):
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)
        with _patch_from_pretrained(model):
            tts.load()
        assert tts._pipeline is not None

    def test_fp16_applies_to_t3_on_cuda(self):
        model = _make_mock_model(device="cuda")
        tts = StreamingTTS(device="cuda", watermark=False)
        with _patch_from_pretrained(model):
            tts.load()
        model.t3.to.assert_called()

    def test_fp16_skipped_on_cpu(self):
        model = _make_mock_model(device="cpu")
        tts = StreamingTTS(device="cpu", watermark=False)
        with _patch_from_pretrained(model):
            tts.load()
        # T3.to() should not be called with dtype on CPU
        for call in model.t3.to.call_args_list:
            assert "dtype" not in (call.kwargs or {}), "FP16 should not be applied on CPU"

    def test_fp16_explicit_override(self):
        model = _make_mock_model(device="cpu")
        tts = StreamingTTS(device="cpu", fp16=True, watermark=False)
        with _patch_from_pretrained(model):
            tts.load()
        model.t3.to.assert_called()


# ---------------------------------------------------------------------------
# 5. synthesize() auto-loads with warning
# ---------------------------------------------------------------------------

class TestSynthesizeAutoLoad:
    async def test_synthesize_auto_loads_with_warning(self):
        """Calling synthesize without load() emits UserWarning and auto-loads."""
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = _mock_pipeline_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                chunks = []
                async for chunk in tts.synthesize("hello"):
                    chunks.append(chunk)

                user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
                assert len(user_warnings) == 1
                assert "auto-loading" in str(user_warnings[0].message).lower()

    async def test_no_warning_if_already_loaded(self):
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = _mock_pipeline_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline):
            tts.load()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                async for _ in tts.synthesize("hello"):
                    pass
                user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
                assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# 6. synthesize() yields AudioChunk objects
# ---------------------------------------------------------------------------

class TestSynthesizeYield:
    async def test_synthesize_yields_audio_chunks(self):
        """Loaded TTS yields AudioChunk objects from pipeline."""
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = _mock_pipeline_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline):
            tts.load()

        chunks = []
        async for chunk in tts.synthesize("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert all(isinstance(c, AudioChunk) for c in chunks)
        assert chunks[-1].is_final is True


# ---------------------------------------------------------------------------
# 7 & 8. Watermark
# ---------------------------------------------------------------------------

class TestWatermark:
    async def test_watermark_true_applies_perth(self):
        """watermark=True applies Perth watermarker to each chunk during synthesize."""
        import numpy as np

        model = _make_mock_model()
        mock_wm = MagicMock()
        mock_wm.apply.return_value = np.zeros(50, dtype=np.float32)

        tts = StreamingTTS(device="cpu", watermark=True)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = _mock_pipeline_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline), \
             patch("chatterstream.streaming_tts._try_load_watermarker",
                   return_value=mock_wm):
            tts.load()

        chunks = []
        async for chunk in tts.synthesize("Hello"):
            chunks.append(chunk)

        # Called once per chunk (2 chunks from mock pipeline)
        assert mock_wm.apply.call_count == 2

    def test_watermark_false_skips_watermarking(self):
        """watermark=False creates no watermarker."""
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        with _patch_from_pretrained(model):
            tts.load()

        assert tts._watermarker is None


# ---------------------------------------------------------------------------
# 9 & 10. Voice routing
# ---------------------------------------------------------------------------

class TestVoiceRouting:
    async def test_voice_builtin_routes_to_builtin_conds(self):
        """voice='builtin' passes __builtin__ sentinel to pipeline."""
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        captured_voices = []

        async def capturing_synthesize(text, voice):
            captured_voices.append(voice)
            yield _make_audio_chunk(0, is_final=True)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = capturing_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline):
            tts.load()

        async for _ in tts.synthesize("Hello", voice="builtin"):
            pass

        assert captured_voices == ["__builtin__"]

    async def test_voice_file_path_routes_to_cache(self):
        """voice='/path/to/file.wav' passes path directly to pipeline."""
        model = _make_mock_model()
        tts = StreamingTTS(device="cpu", watermark=False)

        captured_voices = []

        async def capturing_synthesize(text, voice):
            captured_voices.append(voice)
            yield _make_audio_chunk(0, is_final=True)

        mock_pipeline = MagicMock()
        mock_pipeline.synthesize = capturing_synthesize

        with _patch_from_pretrained(model), \
             patch.object(tts, "_build_pipeline", return_value=mock_pipeline):
            tts.load()

        async for _ in tts.synthesize("Hello", voice="/path/to/file.wav"):
            pass

        assert captured_voices == ["/path/to/file.wav"]


# ---------------------------------------------------------------------------
# _RoutingCondCache (internal component)
# ---------------------------------------------------------------------------

class TestRoutingCondCache:
    def test_builtin_returns_preloaded_conds(self):
        from chatterstream.streaming_tts import _RoutingCondCache

        mock_conds = MagicMock()
        mock_model = MagicMock()
        cache = _RoutingCondCache(mock_conds, mock_model)
        result = cache.get("__builtin__")
        assert result is mock_conds

    def test_file_path_calls_prepare_conditionals(self):
        from chatterstream.streaming_tts import _RoutingCondCache

        mock_conds = MagicMock()
        mock_model = MagicMock()
        expected = MagicMock()
        mock_model.prepare_conditionals.return_value = expected

        cache = _RoutingCondCache(mock_conds, mock_model)
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getmtime", return_value=1234.0):
            result = cache.get("/path/to/voice.wav")

        assert result is expected

    def test_clear_resets_cache(self):
        from chatterstream.streaming_tts import _RoutingCondCache

        cache = _RoutingCondCache(MagicMock(), MagicMock())
        cache._file_cache.clear = MagicMock()
        cache.clear()
        cache._file_cache.clear.assert_called_once()
