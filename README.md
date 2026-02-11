# chatterstream-tts

Streaming TTS wrapper for [Chatterbox](https://github.com/resemble-ai/chatterbox). Turns the monolithic generate-then-play API into an async chunk-by-chunk streaming pipeline with a 3-line interface.

## What is this?

Chatterbox generates high-quality speech but blocks until the entire waveform is ready (1-3s for typical sentences). Chatterstream breaks this into a streaming pipeline that yields audio chunks as they're generated, targeting 300-500ms time-to-first-audio on Apple Silicon.

This is a **personal project** extracted from a Chatterbox fork. It wraps upstream internals (T3 token generation, S3Gen vocoding, HiFiGAN) into a SOLID component pipeline without modifying upstream code.

## Architecture

```
Text Input
    │
    ▼
┌──────────────┐
│TextProcessor │  Regex normalization + GPT-2 tokenization
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ConditioningCache │  Cached voice embeddings (LRU, mtime-aware)
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│TokenGenerator│  T3 autoregressive decoding, yields token chunks
└──────┬───────┘
       │  adaptive chunking (25 tok first, 75 tok subsequent)
       ▼
┌─────────────┐
│ChunkVocoder │  S3Gen causal decode (finalize=False until last)
└──────┬──────┘
       │  flow overlap handling + HiFiGAN source caching
       ▼
┌──────────────────┐
│AudioPostProcessor│  Fade-in, normalization, PCM int16 conversion
└──────┬───────────┘
       │
       ▼
  AsyncIterator[AudioChunk]  →  24 kHz mono int16
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `chatterbox-tts` | 0.1.6 | Upstream TTS model (T3 + S3Gen + HiFiGAN) |
| `torch` | >=2.6.0 | Inference runtime |
| `av` | >=12.0.0 | *Optional* — Opus/HLS media encoding |

**Python**: >=3.10 (tested on 3.11; 3.13+ may have torch compatibility issues)

**Hardware**: Runs on CPU, CUDA, or Apple Silicon (MPS). FP16 is auto-enabled on MPS/CUDA.

## Installation

```bash
# Clone (private repo)
git clone git@github.com:devinmlowe/chatterstream-tts.git
cd chatterstream-tts

# Create venv (Python 3.11 recommended)
uv venv --python python3.11
source .venv/bin/activate  # bash/zsh
# or: source .venv/bin/activate.fish  # fish

# Install with dev dependencies
uv pip install -e ".[dev]"

# Optional: media extras (Opus encoding, HLS segmenting)
uv pip install -e ".[media]"
```

## Usage

### Basic streaming

```python
from chatterstream import StreamingTTS

tts = StreamingTTS()
tts.load()  # loads model weights (~2-5s)

async for chunk in tts.synthesize("Hello world"):
    play(chunk.pcm_bytes)  # 24 kHz mono int16
```

### Custom voice

```python
async for chunk in tts.synthesize("Hello", voice="/path/to/voice.wav"):
    play(chunk.pcm_bytes)
```

### Configuration

```python
from chatterstream import StreamingTTS, StreamConfig

config = StreamConfig(
    first_chunk_tokens=25,       # tokens before first audio yield
    subsequent_chunk_tokens=75,  # tokens per subsequent chunk
    overlap_tokens=0,            # token overlap between chunks
    temperature=0.8,
    top_k=1000,
)

tts = StreamingTTS(device="mps", config=config, watermark=False)
tts.load()
```

### Media encoders

The core pipeline yields raw PCM audio. Two optional encoders (requiring `pip install chatterstream-tts[media]`) handle delivery formats:

**OpusEncoder** — Encodes PCM to OGG/Opus. Best for low-latency delivery over WebSockets or direct streaming where you control both ends. Superior compression at low bitrates, near-zero codec delay. The tradeoff: browsers can't play a raw OGG/Opus stream over plain HTTP — you need JavaScript (e.g. Web Audio API) or a WebSocket to decode it client-side.

**HLSSegmenter** — Encodes PCM to MPEG-TS/AAC segments with an m3u8 playlist. HLS (HTTP Live Streaming) is the standard used by every browser, phone, and smart TV — a plain `<audio>` tag pointed at the m3u8 URL just works, no JavaScript required. Audio is split into small segments (~1-2s each) served over regular HTTP. The tradeoff: segment-based delivery adds inherent latency (the player must buffer at least one segment before playback starts).

These are **complementary, not alternatives**. Use Opus for real-time applications (voice agents, WebSocket APIs) where you control the client. Use HLS when you need universal browser playback with zero client-side code.

The example server at `examples/streaming_server.py` uses HLS because it's the simplest way to test synthesis in any browser — just open the URL and hit play.

## Running tests

```bash
# All unit + integration tests (no model weights needed)
pytest tests/ -v --tb=short

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## License

MIT — same as upstream [Chatterbox](https://github.com/resemble-ai/chatterbox).
