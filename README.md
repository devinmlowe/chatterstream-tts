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
╔═════════════════════════════════════════════════════════════╗
║  chatterstream-tts (streaming wrapper)                      ║
║                                                             ║
║  ┌──────────────┐                                           ║
║  │TextProcessor │  Regex normalization + GPT-2 tokenization ║
║  └──────┬───────┘                                           ║
║         │                                                   ║
║         ▼                                                   ║
║  ┌──────────────────┐                                       ║
║  │ConditioningCache │  Cached voice embeddings (LRU)        ║
║  └──────┬───────────┘                                       ║
║         │                                                   ║
║         ▼                                                   ║
║  ┌──────────────┐                                           ║
║  │TokenGenerator│  Yields token chunks as they generate     ║
║  └──────┬───────┘                                           ║
║         │  adaptive chunking (25 tok first, 75 subsequent)  ║
║         ▼                                                   ║
║  ┌─────────────┐                                            ║
║  │ChunkVocoder │  Per-chunk vocoding with source caching    ║
║  └──────┬──────┘                                            ║
║         │                                                   ║
║         ▼                                                   ║
║  ┌──────────────────┐                                       ║
║  │AudioPostProcessor│  Fade-in, normalize, PCM int16        ║
║  └──────┬───────────┘                                       ║
║         │                                                   ║
╠═══ calls into ══════════════════════════════════════════════╣
║                                                             ║
║  chatterbox-tts (upstream model by Resemble AI)             ║
║                                                             ║
║    T3 (350M)  ─  Autoregressive text-to-token transformer   ║
║    S3Gen      ─  Flow-matching vocoder + HiFiGAN (257M)     ║
║    VoiceEnc   ─  Speaker embedding extraction (LSTM)        ║
║                                                             ║
╚══════════════════════════════════╤══════════════════════════╝
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

**Python**: 3.10 or 3.11 (3.12+ fails due to upstream numpy/distutils incompatibility)

**Hardware**: Runs on CPU, CUDA, or Apple Silicon (MPS).

### FP16 (half-precision)

Neural networks store their weights and perform math using floating-point numbers. The default is FP32 (32-bit / "full precision") — each number uses 32 bits of memory. FP16 (16-bit / "half precision") uses half the memory and runs roughly 2x faster on GPUs that have dedicated half-precision hardware, which includes both NVIDIA CUDA GPUs and Apple Silicon (MPS).

The tradeoff is reduced numerical range, but for TTS inference (as opposed to training) the quality difference is inaudible. Chatterstream auto-enables FP16 on MPS and CUDA because the speed and memory savings are significant with no perceptible quality loss.

On CPU, FP16 is **not** enabled by default — most CPUs lack native half-precision support, so FP16 would actually be *slower* as the CPU emulates it in software. CPU inference uses FP32.

**Note:** Only the T3 text-to-token model runs in FP16. The S3Gen vocoder (HiFiGAN) stays in FP32 because its audio reconstruction is more sensitive to precision, particularly on MPS where dtype mismatches cause errors.

You can override the auto-detection:

```python
# Force FP16 on (even on CPU — not recommended)
tts = StreamingTTS(fp16=True)

# Force FP16 off (even on GPU — full precision, slower)
tts = StreamingTTS(fp16=False)

# Auto-detect (default) — FP16 on MPS/CUDA, FP32 on CPU
tts = StreamingTTS()
```

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

### Hugging Face authentication

Model weights (~600MB) are hosted on Hugging Face at [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) and downloaded automatically the first time you call `.load()`. You need a (free) Hugging Face account and token:

```bash
# Install the Hugging Face CLI (included with chatterbox-tts)
# Then log in — this saves your token locally
huggingface-cli login
```

You can also set the `HF_TOKEN` environment variable instead:

```bash
export HF_TOKEN=hf_your_token_here
```

After the first download, weights are cached locally and no network access is needed for subsequent runs.

## Quick Start

```bash
# 1. Log in to Hugging Face (first time only — needed to download model weights)
huggingface-cli login
```

```python
# 2. Stream speech
from chatterstream import StreamingTTS

tts = StreamingTTS()
tts.load()  # downloads weights on first run (~600MB), then ~2-5s from cache

async for chunk in tts.synthesize("Hello world"):
    play(chunk.pcm_bytes)  # 24 kHz mono int16
```

That's it. The pipeline handles tokenization, chunked generation, vocoding, and audio post-processing internally. Each `AudioChunk` contains raw PCM bytes (24 kHz, mono, int16) plus metadata (`is_final`, `chunk_index`, `duration_ms`).

## Advanced Usage

### Loading the model

Model weights (~600MB) must be loaded before synthesis. There are three ways to handle this, each suited to different scenarios:

**Explicit load** — Call `.load()` yourself before synthesizing. This is the recommended approach for servers and long-running processes. You control exactly when the 2-5 second load happens (e.g. at startup, not on the first user request), and `is_loaded` lets you gate readiness checks.

```python
tts = StreamingTTS()
tts.load()  # 2-5s, blocks until ready
# tts.is_loaded == True
```

**Chained load** — `.load()` returns `self`, so you can construct and load in one line. Convenient for scripts and notebooks where you don't need the intermediate unloaded state.

```python
tts = StreamingTTS(device="mps", watermark=False).load()
```

**Auto-load (lazy)** — If you call `.synthesize()` without loading first, the model loads automatically. A `UserWarning` is emitted so you know it happened. This is fine for quick experiments but not ideal in production — the first synthesis call silently takes an extra 2-5 seconds, which can be confusing.

```python
tts = StreamingTTS()

# First call triggers auto-load (emits: "Model not loaded. Call .load()
# explicitly for faster first synthesis. Auto-loading now...")
async for chunk in tts.synthesize("Hello world"):
    play(chunk.pcm_bytes)

# Subsequent calls are fast — model stays loaded
async for chunk in tts.synthesize("More text"):
    play(chunk.pcm_bytes)
```

**When to use which:**

| Approach | Best for | Tradeoff |
|---|---|---|
| Explicit `.load()` | Servers, APIs, anything with a startup phase | You manage the load timing |
| Chained `.load()` | Scripts, notebooks, one-off experiments | No access to unloaded state |
| Auto-load | Quick prototyping, REPL exploration | Surprising latency on first call |

### Custom voice

```python
async for chunk in tts.synthesize("Hello", voice="/path/to/voice.wav"):
    play(chunk.pcm_bytes)
```

Voice files are cached by path with mtime-based invalidation — re-encoding the same file at the same path busts the cache automatically.

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

The core pipeline yields raw PCM audio (24 kHz, mono, int16). To deliver that audio to clients, you need an encoding/transport layer. Two optional encoders (requiring `pip install chatterstream-tts[media]`) handle this:

**OpusEncoder** — Encodes PCM to OGG/Opus. Best for low-latency delivery over WebSockets or direct streaming where you control both ends. Superior compression at low bitrates, near-zero codec delay. The tradeoff: browsers can't play a raw OGG/Opus stream over plain HTTP — you need JavaScript (e.g. Web Audio API) or a WebSocket to decode it client-side.

```python
from chatterstream.opus_encoder import OpusEncoder

encoder = OpusEncoder(input_sample_rate=24000, bitrate=64000)

async for chunk in tts.synthesize("Hello world"):
    ogg_bytes = encoder.encode(chunk.pcm_bytes)
    websocket.send(ogg_bytes)  # stream over WebSocket

# Flush the encoder when done (emits final OGG pages)
final_bytes = encoder.finalize()
websocket.send(final_bytes)

# Reset for the next utterance (reuses the encoder object)
encoder.reset()
```

**HLSSegmenter** — Encodes PCM to MPEG-TS/AAC segments with an m3u8 playlist. HLS (HTTP Live Streaming) is the standard used by every browser, phone, and smart TV — a plain `<audio>` tag pointed at the m3u8 URL just works, no JavaScript required. Audio is split into small segments (~1-2s each) served over regular HTTP. The tradeoff: segment-based delivery adds inherent latency (the player must buffer at least one segment before playback starts).

```python
from chatterstream.hls_segmenter import HLSSegmenter

segmenter = HLSSegmenter(sample_rate=24000, bitrate=96000)

async for chunk in tts.synthesize("Hello world"):
    seg_bytes = segmenter.add_segment(chunk.pcm_bytes)
    # seg_bytes is a self-contained MPEG-TS segment
    # serve it at /seg{index}.ts

# Flush encoder and mark the stream complete
segmenter.finalize()

# Generate the m3u8 playlist (references seg0.ts, seg1.ts, ...)
playlist = segmenter.playlist()
# serve playlist at /stream.m3u8

# Retrieve any segment by index
segment_0 = segmenter.get_segment(0)
```

**These are complementary, not alternatives.** Use Opus for real-time applications (voice agents, WebSocket APIs) where you control the client. Use HLS when you need universal browser playback with zero client-side code. You could even use both — Opus for a native app client and HLS for a web fallback.

See `examples/streaming_server.py` for a complete aiohttp server that uses HLS to serve audio playable in any browser.

## Responsible AI: PerTh Watermarking

Chatterbox includes [Resemble AI's PerTh (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) — imperceptible neural watermarks embedded in generated audio that survive MP3 compression, editing, and common manipulations while maintaining nearly 100% detection accuracy. This is an important tool for responsible use of synthetic speech.

Chatterstream supports PerTh watermarking and **enables it by default**:

```python
# Watermarking on (default)
tts = StreamingTTS(watermark=True)

# Watermarking off (for local development / testing only)
tts = StreamingTTS(watermark=False)
```

Each audio chunk is watermarked as it passes through the pipeline, so streaming delivery doesn't bypass the watermark the way a naive chunk-by-chunk approach might.

**Note:** The open-source `resemble-perth` package ships with `PerthImplicitWatermarker` disabled (set to `None`). Chatterstream detects this and falls back gracefully — watermarking is silently skipped with a log message. If you have access to a full PerTh implementation, it activates automatically.

### Extracting watermarks

You can verify whether audio was generated by Chatterbox/Chatterstream using the PerTh detector:

```python
import perth
import librosa

watermarked_audio, sr = librosa.load("output.wav", sr=None)

watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```

Watermarking is an important part of the responsible deployment of speech synthesis technology. If you distribute audio generated by Chatterstream, you should leave watermarking enabled so that synthetic speech remains identifiable.

## Running tests

```bash
# All unit + integration tests (no model weights needed)
pytest tests/ -v --tb=short

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## License

MIT — same as upstream [Chatterbox](https://github.com/resemble-ai/chatterbox).
