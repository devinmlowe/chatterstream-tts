# chatterstream-tts — LLM Integration Guide

This document provides instructions for LLMs and AI agents on how to use the `chatterstream-tts` Python library to generate streaming speech audio from text.

## What This Library Does

`chatterstream-tts` is a streaming wrapper around [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI. It converts text to speech audio in real-time, yielding chunks of audio as they are generated rather than waiting for the entire utterance to complete.

**Output format**: 24 kHz, mono, 16-bit PCM audio delivered as an async stream of `AudioChunk` objects.

## Prerequisites

Before using this library, the environment must have:

1. **Python 3.10 or 3.11** (3.12+ is incompatible due to upstream numpy constraints)
2. **The package installed**: `pip install chatterstream-tts`
3. **Hugging Face authentication**: Model weights (~600MB) are downloaded from Hugging Face Hub on first use. Either:
   - Run `huggingface-cli login` interactively, OR
   - Set the `HF_TOKEN` environment variable to a valid Hugging Face token
4. **Hardware**: CPU works but is slow. Apple Silicon (MPS) or NVIDIA GPU (CUDA) recommended. FP16 is auto-enabled on GPU hardware.

After the first model download, weights are cached locally and no network access is needed.

## Core API

### Import

```python
from chatterstream import StreamingTTS, StreamConfig, AudioChunk
```

### Constructor

```python
StreamingTTS(
    device: str | None = None,     # "cpu", "cuda", "mps", or None (auto-detect)
    config: StreamConfig | None = None,  # pipeline tuning parameters
    watermark: bool = True,        # enable PerTh audio watermarking
    fp16: bool | None = None,      # None=auto (GPU→FP16, CPU→FP32)
)
```

### Loading the Model

You MUST load the model before synthesis. There are three patterns:

```python
# Pattern 1: Explicit load (RECOMMENDED for servers/agents)
tts = StreamingTTS()
tts.load()  # blocks for 2-5 seconds

# Pattern 2: Chained load (convenient one-liner)
tts = StreamingTTS(device="mps", watermark=False).load()

# Pattern 3: Auto-load (emits UserWarning, not recommended)
# Calling synthesize() without load() triggers automatic loading.
# Avoid this — it creates unpredictable latency on the first call.
```

Check if loaded: `tts.is_loaded` returns `bool`.

### Synthesizing Speech

`synthesize()` is an async generator. You MUST use it inside an async context.

```python
async for chunk in tts.synthesize("Text to speak"):
    # chunk is an AudioChunk with these fields:
    #   chunk.pcm_bytes    — bytes, raw PCM int16 audio
    #   chunk.sample_rate  — int, always 24000
    #   chunk.is_final     — bool, True on the last chunk
    #   chunk.chunk_index  — int, 0-based sequential index
    #   chunk.duration_ms  — float, duration of this chunk in milliseconds
    do_something_with(chunk.pcm_bytes)
```

### Custom Voice

Pass a path to a `.wav` file to clone a voice:

```python
async for chunk in tts.synthesize("Hello", voice="/path/to/voice.wav"):
    ...
```

Use `voice="builtin"` (the default) for the model's built-in voice. Voice conditioning is cached by file path with mtime-based invalidation.

### StreamConfig Parameters

```python
StreamConfig(
    first_chunk_tokens=25,       # tokens before first audio yield (~1s)
    subsequent_chunk_tokens=75,  # tokens per subsequent chunk (~3s)
    overlap_tokens=0,            # token overlap between chunks (0 recommended)
    strategy=ChunkStrategy.ADAPTIVE,  # or ChunkStrategy.SENTENCE_ALIGNED
    temperature=0.8,             # T3 sampling temperature
    top_k=1000,                  # T3 top-k sampling
    top_p=0.95,                  # T3 nucleus sampling
    repetition_penalty=1.2,      # T3 repetition penalty
    max_gen_len=1000,            # maximum generation length in tokens
)
```

## Common Integration Patterns

### Save to WAV File

```python
import wave

tts = StreamingTTS(watermark=False).load()

pcm_data = bytearray()
async for chunk in tts.synthesize("Hello world"):
    pcm_data.extend(chunk.pcm_bytes)

with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(24000)
    wf.writeframes(bytes(pcm_data))
```

### Stream Over WebSocket (Opus)

Requires: `pip install chatterstream-tts[media]`

```python
from chatterstream.opus_encoder import OpusEncoder

encoder = OpusEncoder(input_sample_rate=24000, bitrate=64000)

async for chunk in tts.synthesize("Hello world"):
    ogg_bytes = encoder.encode(chunk.pcm_bytes)
    await websocket.send(ogg_bytes)

final_bytes = encoder.finalize()
await websocket.send(final_bytes)
encoder.reset()  # reuse for next utterance
```

### Serve via HLS (Browser Playback)

Requires: `pip install chatterstream-tts[media]`

```python
from chatterstream.hls_segmenter import HLSSegmenter

segmenter = HLSSegmenter(sample_rate=24000, bitrate=96000)

async for chunk in tts.synthesize("Hello world"):
    seg_bytes = segmenter.add_segment(chunk.pcm_bytes)
    # Store seg_bytes, serve at /seg{segmenter.segment_count - 1}.ts

segmenter.finalize()
playlist_m3u8 = segmenter.playlist()  # serve at /playlist.m3u8
segment_data = segmenter.get_segment(0)  # retrieve by index
```

### Multiple Sequential Utterances

The `StreamingTTS` instance is reusable. Load once, synthesize many times:

```python
tts = StreamingTTS().load()

for text in ["First sentence.", "Second sentence.", "Third sentence."]:
    async for chunk in tts.synthesize(text):
        process(chunk)
```

## Important Constraints

1. **Async only**: `synthesize()` is an async generator. It requires `asyncio` or an equivalent async runtime. You cannot use it in synchronous code without `asyncio.run()` or similar.

2. **Single-threaded inference**: The underlying model is not thread-safe. Do not call `synthesize()` concurrently from multiple tasks without a lock:
   ```python
   lock = asyncio.Lock()
   async with lock:
       async for chunk in tts.synthesize(text):
           ...
   ```

3. **Memory**: The model uses ~1-2GB of GPU/system memory when loaded. It stays in memory for the lifetime of the `StreamingTTS` instance.

4. **First-run download**: The very first `.load()` call downloads ~600MB of model weights from Hugging Face Hub. This requires network access and HF authentication. Subsequent loads use the local cache.

5. **No batch inference**: This library targets single-utterance streaming. It does not support batching multiple texts in parallel.

6. **Audio format is fixed**: Output is always 24 kHz, mono, 16-bit signed integer PCM. Use the media encoders (OpusEncoder, HLSSegmenter) to transcode if needed.

## Error Handling

| Error | Cause | Fix |
|---|---|---|
| `OSError: HF_TOKEN` / 401 | Hugging Face not authenticated | Run `huggingface-cli login` or set `HF_TOKEN` |
| `FileNotFoundError: Voice file not found` | Invalid voice path | Check the path exists and is a `.wav` file |
| `RuntimeError: MPS backend` | MPS dtype issues | Let the library auto-detect FP16, or use `fp16=False` |
| `ModuleNotFoundError: av` | PyAV not installed | Run `pip install chatterstream-tts[media]` |
| `UserWarning: Model not loaded` | Called `synthesize()` before `load()` | Call `.load()` explicitly before synthesis |

## Package Structure

```
chatterstream-tts
├── chatterstream              # import name
│   ├── StreamingTTS           # primary API (from chatterstream import StreamingTTS)
│   ├── AudioChunk             # output data type
│   ├── StreamConfig           # pipeline configuration
│   ├── TokenChunk             # internal token data type
│   ├── ChunkStrategy          # ADAPTIVE or SENTENCE_ALIGNED enum
│   ├── opus_encoder           # optional: from chatterstream.opus_encoder import OpusEncoder
│   └── hls_segmenter          # optional: from chatterstream.hls_segmenter import HLSSegmenter
└── chatterbox-tts             # upstream dependency (installed automatically)
```

## Minimal Working Example

```python
import asyncio
from chatterstream import StreamingTTS

async def main():
    tts = StreamingTTS(watermark=False)
    tts.load()

    all_audio = bytearray()
    async for chunk in tts.synthesize("The quick brown fox jumps over the lazy dog."):
        all_audio.extend(chunk.pcm_bytes)
        print(f"Chunk {chunk.chunk_index}: {chunk.duration_ms:.0f}ms, final={chunk.is_final}")

    print(f"Total audio: {len(all_audio)} bytes ({len(all_audio) / 2 / 24000:.1f}s)")

asyncio.run(main())
```
