"""Minimal HLS streaming TTS server using the StreamingTTS facade.

Run:  python examples/streaming_server.py
Open: http://127.0.0.1:8877/

HLS streaming pattern:
  GET /hls/{sid}/playlist.m3u8?text=...  -> live m3u8 (creates session on first hit)
  GET /hls/{sid}/seg{n}.ts               -> MPEG-TS/AAC segment
  Client: Safari native HLS or hls.js fallback
"""

import asyncio
import time
import logging

from aiohttp import web

from chatterstream import StreamingTTS
from chatterstream.hls_segmenter import HLSSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model (lazy-loaded on startup)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000  # Chatterbox native sample rate

tts: StreamingTTS | None = None


def _load_model():
    global tts
    if tts is not None:
        return

    logger.info("Loading model via StreamingTTS facade...")
    tts = StreamingTTS(watermark=False)
    tts.load()
    logger.info("Model loaded — ready for requests")

    # Pre-warm: single inference to trigger kernel compilation / lazy allocs
    logger.info("Pre-warming pipeline...")
    warmup_start = time.monotonic()

    async def _warmup():
        async for _ in tts.synthesize("Hello, this is a warmup sentence."):
            pass

    asyncio.run(_warmup())
    logger.info(f"Pre-warm done in {time.monotonic() - warmup_start:.1f}s")


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Streaming TTS Test</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #121212;
    color: #fff;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    padding-top: max(20px, env(safe-area-inset-top));
    padding-bottom: max(20px, env(safe-area-inset-bottom));
  }
  h1 {
    font-size: 18px;
    font-weight: 600;
    color: #B3B3B3;
    letter-spacing: -0.02em;
    margin-bottom: 24px;
  }
  .container {
    width: 100%;
    max-width: 440px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    flex: 1;
  }
  textarea {
    width: 100%;
    min-height: 100px;
    background: #282828;
    border: 1px solid #333;
    border-radius: 12px;
    color: #fff;
    font-size: 16px;
    padding: 14px;
    resize: vertical;
    font-family: inherit;
    line-height: 1.5;
    transition: border-color 0.2s;
  }
  textarea:focus {
    outline: none;
    border-color: #1DB954;
  }
  textarea::placeholder { color: #6A6A6A; }
  button {
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 500px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  button.send {
    background: #1DB954;
    color: #fff;
  }
  button.send:active { background: #1ed760; }
  button.send:disabled {
    background: #333;
    color: #6A6A6A;
    cursor: not-allowed;
  }
  .timer-box {
    background: #181818;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
  .timer-label {
    font-size: 12px;
    font-weight: 600;
    color: #6A6A6A;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .timer-value {
    font-size: 48px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    color: #B3B3B3;
    transition: color 0.2s;
  }
  .timer-value.running { color: #1DB954; }
  .timer-value.stopped { color: #fff; }
  .status {
    font-size: 13px;
    color: #6A6A6A;
    text-align: center;
    min-height: 20px;
  }
  .status.active { color: #B3B3B3; }
  .status.error { color: #E91429; }
  .stats {
    background: #181818;
    border-radius: 12px;
    padding: 14px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .stat { text-align: center; }
  .stat-val {
    font-size: 20px;
    font-weight: 600;
    color: #fff;
    font-variant-numeric: tabular-nums;
  }
  .stat-lbl {
    font-size: 11px;
    color: #6A6A6A;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 2px;
  }
</style>
</head>
<body>
  <div class="container">
    <h1>Streaming TTS Test</h1>
    <textarea id="text" placeholder="Type something to speak..."
      >Hello, this is a test of the streaming pipeline.</textarea>
    <button class="send" id="send" onclick="run()">Send</button>
    <audio id="player" preload="none" style="display:none"></audio>
    <div class="timer-box">
      <div class="timer-label">Time to first audio</div>
      <div class="timer-value" id="timer">0.000s</div>
    </div>
    <div class="stats" id="stats" style="display:none">
      <div class="stat">
        <div class="stat-val" id="serverMs">-</div>
        <div class="stat-lbl">Server ready</div>
      </div>
      <div class="stat">
        <div class="stat-val" id="segments">-</div>
        <div class="stat-lbl">Segments</div>
      </div>
    </div>
    <div class="status" id="status">Ready</div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js"></script>
<script>
const timerEl = document.getElementById('timer');
const statusEl = document.getElementById('status');
const sendBtn = document.getElementById('send');
const textEl = document.getElementById('text');
const statsEl = document.getElementById('stats');
const serverMsEl = document.getElementById('serverMs');
const segmentsEl = document.getElementById('segments');
const player = document.getElementById('player');

let timerRAF = null;
let startTime = 0;
let activeHls = null;

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = 'status ' + (cls || '');
}

function startTimer() {
  startTime = performance.now();
  timerEl.className = 'timer-value running';
  function tick() {
    const ms = performance.now() - startTime;
    timerEl.textContent = (ms / 1000).toFixed(3) + 's';
    timerRAF = requestAnimationFrame(tick);
  }
  tick();
}

function stopTimer() {
  if (timerRAF) cancelAnimationFrame(timerRAF);
  timerRAF = null;
  const ms = performance.now() - startTime;
  timerEl.textContent = (ms / 1000).toFixed(3) + 's';
  timerEl.className = 'timer-value stopped';
}

function cleanup() {
  if (activeHls) {
    activeHls.destroy();
    activeHls = null;
  }
}

async function run() {
  const text = textEl.value.trim();
  if (!text) return;

  cleanup();
  sendBtn.disabled = true;
  statsEl.style.display = 'none';
  setStatus('Generating...', 'active');
  startTimer();

  const sid = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const base = location.pathname.endsWith('/') ? location.pathname : location.pathname + '/';
  const playlistUrl = base + 'hls/' + sid + '/playlist.m3u8?text=' + encodeURIComponent(text);

  player.onplaying = () => {
    stopTimer();
    setStatus('Playing...', 'active');
  };

  player.onended = () => {
    cleanup();
    setStatus('Done', '');
    sendBtn.disabled = false;
  };

  player.onerror = () => {
    cleanup();
    stopTimer();
    const err = player.error;
    setStatus('Playback error: ' + (err ? err.message : 'unknown'), 'error');
    sendBtn.disabled = false;
  };

  function fetchStats() {
    fetch(playlistUrl).then(r => {
      if (!r.ok) return;
      const srvMs = r.headers.get('X-First-Segment-Ms');
      const segCount = r.headers.get('X-Segment-Count');
      if (srvMs || segCount) {
        statsEl.style.display = 'grid';
        serverMsEl.textContent = srvMs ? (srvMs / 1000).toFixed(2) + 's' : '-';
        segmentsEl.textContent = segCount || '-';
      }
    }).catch(() => {});
  }
  player.addEventListener('playing', fetchStats, {once: true});

  if (player.canPlayType('application/vnd.apple.mpegurl')) {
    player.src = playlistUrl;
    try { await player.play(); } catch (e) {
      stopTimer();
      setStatus('Play blocked: ' + e.message, 'error');
      sendBtn.disabled = false;
    }
  } else if (typeof Hls !== 'undefined' && Hls.isSupported()) {
    const hls = new Hls({
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 3,
      enableWorker: false,
    });
    activeHls = hls;
    hls.loadSource(playlistUrl);
    hls.attachMedia(player);
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      player.play().catch(e => {
        stopTimer();
        setStatus('Play blocked: ' + e.message, 'error');
        sendBtn.disabled = false;
      });
    });
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) {
        stopTimer();
        setStatus('HLS error: ' + data.details, 'error');
        sendBtn.disabled = false;
        hls.destroy();
        activeHls = null;
      }
    });
  } else {
    setStatus('HLS not supported in this browser', 'error');
    sendBtn.disabled = false;
  }
}

textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    run();
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HLS session management
# ---------------------------------------------------------------------------

_synth_lock: asyncio.Lock | None = None  # initialized in on_startup
_sessions: dict[str, dict] = {}

SESSION_TIMEOUT = 60

# Minimum PCM bytes before flushing a segment (0.5s at 24kHz mono int16).
MIN_SEGMENT_BYTES = 24000  # 12000 samples * 2 bytes/sample

# Silence primer: 1s of silence primes AAC encoder and gives Safari a
# segment to buffer while synthesis runs in the background.
SILENCE_PRIMER_SAMPLES = 24000  # 1.0s at 24kHz


async def _generate(sid: str, text: str) -> None:
    """Background task: synthesize text and encode PCM to HLS segments."""
    session = _sessions.get(sid)
    if not session:
        return

    segmenter: HLSSegmenter = session["segmenter"]
    start = time.monotonic()
    chunk_count = 0
    pcm_buffer = b""

    # Silence primer: primes AAC encoder, gives Safari immediate content
    silence_pcm = b"\x00" * (SILENCE_PRIMER_SAMPLES * 2)
    segmenter.add_segment(silence_pcm)
    primer_ms = SILENCE_PRIMER_SAMPLES / SAMPLE_RATE * 1000
    logger.info(f"  [{sid[:8]}] silence primer: {primer_ms:.0f}ms")

    # Signal ready — Safari can buffer silence while synthesis runs
    session["first_seg_ms"] = (time.monotonic() - start) * 1000
    session["ready"].set()
    logger.info(
        f"  [{sid[:8]}] playlist ready: "
        f"{session['first_seg_ms']:.0f}ms (silence only)"
    )

    try:
        async with _synth_lock:
            async for chunk in tts.synthesize(text):
                chunk_count += 1
                pcm_buffer += chunk.pcm_bytes

                if len(pcm_buffer) >= MIN_SEGMENT_BYTES:
                    seg_samples = len(pcm_buffer) // 2
                    segmenter.add_segment(pcm_buffer)
                    pcm_buffer = b""

                    logger.info(
                        f"  [{sid[:8]}] chunk {chunk.chunk_index}: "
                        f"seg={segmenter.segment_count} segs  "
                        f"audio={seg_samples / SAMPLE_RATE * 1000:.0f}ms  "
                        f"elapsed={time.monotonic() - start:.3f}s"
                    )

    except Exception as e:
        logger.error(f"  [{sid[:8]}] synthesis error: {e}")
    finally:
        # Flush remaining PCM as final segment
        if pcm_buffer:
            segmenter.add_segment(pcm_buffer)
            logger.info(
                f"  [{sid[:8]}] tail merged: "
                f"{len(pcm_buffer) // 2 / SAMPLE_RATE * 1000:.0f}ms into seg "
                f"{segmenter.segment_count - 1}"
            )

        segmenter.finalize()
        session["done"] = True
        if not session["ready"].is_set():
            session["first_seg_ms"] = (time.monotonic() - start) * 1000
            session["ready"].set()
        total = time.monotonic() - start
        logger.info(
            f"  [{sid[:8]}] done: {chunk_count} chunks -> "
            f"{segmenter.segment_count} segs in {total:.3f}s"
        )


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_index(request):
    return web.Response(text=HTML, content_type="text/html")


async def handle_playlist(request):
    """GET /hls/{sid}/playlist.m3u8?text=... — live m3u8 playlist."""
    sid = request.match_info["sid"]
    text = request.query.get("text", "").strip()

    if sid not in _sessions and text:
        segmenter = HLSSegmenter(sample_rate=SAMPLE_RATE, bitrate=96000)
        _sessions[sid] = {
            "segmenter": segmenter,
            "ready": asyncio.Event(),
            "done": False,
            "text": text,
            "first_seg_ms": 0,
        }
        asyncio.create_task(_generate(sid, text))
        logger.info(f"Session {sid[:8]} created for: {text!r}")

    session = _sessions.get(sid)
    if not session:
        return web.Response(status=404, text="Unknown session")

    try:
        await asyncio.wait_for(session["ready"].wait(), timeout=30)
    except asyncio.TimeoutError:
        return web.Response(status=504, text="Synthesis timeout")

    playlist = session["segmenter"].playlist()
    headers = {
        "Cache-Control": "no-cache, no-store",
        "X-First-Segment-Ms": str(int(session.get("first_seg_ms", 0))),
        "X-Segment-Count": str(session["segmenter"].segment_count),
    }
    return web.Response(
        text=playlist,
        content_type="application/vnd.apple.mpegurl",
        headers=headers,
    )


async def handle_segment(request):
    """GET /hls/{sid}/seg{n}.ts — serve MPEG-TS segment."""
    sid = request.match_info["sid"]
    seg_str = request.match_info["seg"]

    session = _sessions.get(sid)
    if not session:
        return web.Response(status=404, text="Unknown session")

    try:
        seg_index = int(seg_str)
    except ValueError:
        return web.Response(status=400, text="Invalid segment index")

    data = session["segmenter"].get_segment(seg_index)
    if data is None:
        return web.Response(status=404, text="Segment not ready")

    return web.Response(
        body=data,
        content_type="video/mp2t",
        headers={"Cache-Control": "max-age=86400"},
    )


async def _cleanup_sessions() -> None:
    """Periodically remove stale sessions."""
    while True:
        await asyncio.sleep(SESSION_TIMEOUT)
        stale = [sid for sid, s in _sessions.items() if s["done"]]
        for sid in stale:
            _sessions.pop(sid, None)
            logger.info(f"  session {sid[:8]} cleaned up")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _load_model()

    app = web.Application()
    # Routes for both direct access and tailscale serve prefix
    for prefix in ("", "/streaming-test"):
        app.router.add_get(prefix + "/", handle_index)
        app.router.add_get(prefix + "/hls/{sid}/playlist.m3u8", handle_playlist)
        app.router.add_get(prefix + "/hls/{sid}/seg{seg}.ts", handle_segment)

    async def on_startup(app):
        global _synth_lock
        _synth_lock = asyncio.Lock()
        asyncio.create_task(_cleanup_sessions())

    app.on_startup.append(on_startup)
    web.run_app(app, host="127.0.0.1", port=8877)


if __name__ == "__main__":
    main()
