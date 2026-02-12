"""StreamingTTS — the primary public API for chatterstream.

Usage::

    from chatterstream import StreamingTTS

    tts = StreamingTTS()
    tts.load()

    async for chunk in tts.synthesize("Hello world"):
        play(chunk.pcm_bytes)  # 24 kHz mono int16
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import AsyncIterator, Optional

import torch

from .audio_post_processor import AudioPostProcessor
from .chunk_vocoder import ChunkVocoder
from .interfaces import ConditioningCacheBase
from .pipeline import StreamingPipeline
from .text_processor import TextProcessor
from .token_generator import TokenGenerator
from .types import AudioChunk, StreamConfig

logger = logging.getLogger(__name__)

# Lazy import: only resolve at load() time so the module is importable
# without chatterbox-tts installed (for e.g. docs / type-checking).
ChatterboxTurboTTS = None  # populated by _ensure_upstream()


def _ensure_upstream():
    global ChatterboxTurboTTS
    if ChatterboxTurboTTS is None:
        # Patch perth stub: the open-source resemble-perth ships
        # PerthImplicitWatermarker as None.  Upstream __init__ calls it
        # unconditionally, so swap in DummyWatermarker to avoid TypeError.
        import perth
        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

        from chatterbox.tts_turbo import ChatterboxTurboTTS as _Cls
        ChatterboxTurboTTS = _Cls


def _try_load_watermarker():
    """Return a Perth watermarker or *None* if perth is unavailable."""
    try:
        import perth
        wm_cls = perth.PerthImplicitWatermarker
        if wm_cls is None or wm_cls is perth.DummyWatermarker:
            logger.info("Perth real watermarker not available — watermarking disabled")
            return None
        return wm_cls()
    except (ImportError, TypeError):
        logger.warning("perth not available — watermarking disabled")
        return None


# ---------------------------------------------------------------------------
# Internal: routing conditioning cache
# ---------------------------------------------------------------------------

class _RoutingCondCache(ConditioningCacheBase):
    """Routes ``"__builtin__"`` to preloaded conds, file paths to the model.

    This keeps the ``StreamingPipeline`` unaware of the special builtin
    voice sentinel while allowing ``StreamingTTS.synthesize(voice="builtin")``
    to skip file I/O entirely.
    """

    def __init__(self, builtin_conds, model, maxsize: int = 8):
        self._builtin = builtin_conds
        self._model = model
        self._maxsize = maxsize
        self._file_cache: OrderedDict[tuple[str, float], object] = OrderedDict()

    # -- ConditioningCacheBase interface ------------------------------------

    def get(self, voice_path: str):
        if voice_path == "__builtin__":
            return self._builtin

        resolved = str(Path(voice_path).resolve())
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        mtime = os.path.getmtime(resolved)
        key = (resolved, mtime)

        if key in self._file_cache:
            self._file_cache.move_to_end(key)
            return self._file_cache[key]

        # Evict stale entries for same path
        stale = [k for k in self._file_cache if k[0] == resolved]
        for k in stale:
            del self._file_cache[k]

        while len(self._file_cache) >= self._maxsize:
            self._file_cache.popitem(last=False)

        result = self._model.prepare_conditionals(resolved)
        self._file_cache[key] = result
        return result

    def clear(self) -> None:
        self._file_cache.clear()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class StreamingTTS:
    """High-level streaming TTS facade.

    Wraps model loading, FP16 optimisation, pipeline wiring, and optional
    Perth watermarking behind a three-line API.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[StreamConfig] = None,
        watermark: bool = True,
        fp16: Optional[bool] = None,
    ):
        self._device_request = device
        self._config = config or StreamConfig()
        self._watermark = watermark
        self._fp16_request = fp16

        # Populated by load()
        self._model = None
        self._pipeline: Optional[StreamingPipeline] = None
        self._watermarker = None
        self._device: Optional[str] = None

    # -- properties ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None

    # -- load ---------------------------------------------------------------

    def load(self) -> "StreamingTTS":
        """Load model weights.  Returns *self* for chaining."""
        _ensure_upstream()

        # 1. Resolve device
        self._device = self._resolve_device()

        # 2. Load model
        self._model = ChatterboxTurboTTS.from_pretrained(self._device)

        # 3. FP16 for T3 only (S3Gen stays FP32 — MPS HiFiGAN dtype issues)
        if self._should_use_fp16():
            t3_dtype = torch.float16
            # Delete unused text embedding table before dtype cast
            if hasattr(self._model.t3, "tfmr") and hasattr(self._model.t3.tfmr, "wte"):
                del self._model.t3.tfmr.wte
            self._model.t3.to(dtype=t3_dtype)
            # Cast T3 conditioning tensors to match T3 dtype
            # (S3Gen conds stay FP32 to match S3Gen)
            if hasattr(self._model, "conds") and self._model.conds is not None:
                t3_cond = getattr(self._model.conds, "t3", None)
                if t3_cond is not None:
                    for attr_name in ("speaker_emb", "emotion_adv"):
                        v = getattr(t3_cond, attr_name, None)
                        if v is not None and torch.is_tensor(v) and v.is_floating_point():
                            setattr(t3_cond, attr_name, v.to(dtype=t3_dtype))

        # 4. Build streaming pipeline
        self._pipeline = self._build_pipeline()

        # 5. Watermarker
        if self._watermark:
            self._watermarker = _try_load_watermarker()
        else:
            self._watermarker = None

        return self

    # -- synthesize ---------------------------------------------------------

    async def synthesize(
        self, text: str, voice: str = "builtin"
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks for *text*.

        Auto-loads with a warning if :meth:`load` has not been called.
        """
        if not self.is_loaded:
            warnings.warn(
                "Model not loaded. Call .load() explicitly for faster first "
                "synthesis. Auto-loading now...",
                UserWarning,
                stacklevel=2,
            )
            self.load()

        voice_path = self._resolve_voice(voice)

        async for chunk in self._pipeline.synthesize(text, voice_path):
            if self._watermarker is not None:
                chunk = self._apply_watermark(chunk)
            yield chunk

    # -- internal helpers ---------------------------------------------------

    def _resolve_device(self) -> str:
        if self._device_request is not None:
            return self._device_request
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _should_use_fp16(self) -> bool:
        if self._fp16_request is not None:
            return self._fp16_request
        return self._device in ("mps", "cuda")

    def _build_pipeline(self) -> StreamingPipeline:
        model = self._model
        cond_cache = _RoutingCondCache(
            builtin_conds=model.conds,
            model=model,
        )
        return StreamingPipeline(
            text_processor=TextProcessor(model.tokenizer),
            conditioning_cache=cond_cache,
            token_generator=TokenGenerator(model.t3, self._config),
            chunk_vocoder=ChunkVocoder(model.s3gen),
            audio_post_processor=AudioPostProcessor(sample_rate=model.sr),
            config=self._config,
            sample_rate=model.sr,
            device=model.device,
        )

    @staticmethod
    def _resolve_voice(voice: str) -> str:
        if voice == "builtin":
            return "__builtin__"
        return voice

    def _apply_watermark(self, chunk: AudioChunk) -> AudioChunk:
        import numpy as np

        pcm = (
            np.frombuffer(chunk.pcm_bytes, dtype=np.int16)
            .astype(np.float32) / 32767.0
        )
        marked = self._watermarker.apply_watermark(pcm, sample_rate=chunk.sample_rate)
        marked_int16 = (marked * 32767).astype(np.int16)

        return AudioChunk(
            pcm_bytes=marked_int16.tobytes(),
            sample_rate=chunk.sample_rate,
            is_final=chunk.is_final,
            chunk_index=chunk.chunk_index,
            duration_ms=chunk.duration_ms,
        )
