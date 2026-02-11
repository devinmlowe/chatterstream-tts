"""Chatterstream — streaming TTS wrapper for Chatterbox.

Core exports::

    from chatterstream import StreamingTTS, AudioChunk, StreamConfig

Media extras (require ``pip install chatterstream-tts[media]``)::

    from chatterstream.opus_encoder import OpusEncoder
    from chatterstream.hls_segmenter import HLSSegmenter
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("chatterstream-tts")
except Exception:
    __version__ = "0.0.0+dev"

from .streaming_tts import StreamingTTS
from .types import AudioChunk, ChunkStrategy, StreamConfig, TokenChunk
from .interfaces import (
    AudioPostProcessorBase,
    ChunkVocoderBase,
    ConditioningCacheBase,
    StreamingPipelineBase,
    TextProcessorBase,
    TokenGeneratorBase,
)

# OpusEncoder and HLSSegmenter NOT imported here — they require PyAV (av).
# Users who need media extras import directly:
#   from chatterstream.opus_encoder import OpusEncoder
#   from chatterstream.hls_segmenter import HLSSegmenter

__all__ = [
    # Primary API
    "StreamingTTS",
    # Data types
    "AudioChunk",
    "ChunkStrategy",
    "StreamConfig",
    "TokenChunk",
    # Interfaces (for extensibility)
    "AudioPostProcessorBase",
    "ChunkVocoderBase",
    "ConditioningCacheBase",
    "StreamingPipelineBase",
    "TextProcessorBase",
    "TokenGeneratorBase",
]
