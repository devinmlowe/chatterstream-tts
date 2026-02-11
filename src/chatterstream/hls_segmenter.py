"""HLS segmenter: PCM int16 bytes to MPEG-TS/AAC segments with m3u8 playlist.

Uses a persistent AAC encoder across segments so the MDCT windowing
produces proper overlap at segment boundaries — eliminating the audible
"garble" artifacts that independent per-segment encoding creates.
"""

from __future__ import annotations

import io
import math

import av


class HLSSegmenter:
    """Encodes PCM int16 mono audio to MPEG-TS/AAC segments for HLS.

    A single AAC encoder runs for the lifetime of the segmenter.
    Each add_segment() call encodes PCM through the persistent encoder
    and captures the new MPEG-TS bytes as a segment.  Because the encoder
    state is continuous, MDCT overlap between frames at segment boundaries
    is handled naturally — no audible artifacts.

    PAT/PMT headers are inserted every 20ms so each segment remains
    independently parseable by HLS players.
    """

    def __init__(self, sample_rate: int = 24000, bitrate: int = 96000):
        self._sample_rate = sample_rate
        self._bitrate = bitrate
        self._segments: list[bytes] = []
        self._durations: list[float] = []
        self._finalized = False

        # Persistent MPEG-TS container + AAC encoder
        self._buf = io.BytesIO()
        self._container = av.open(
            self._buf, "w", format="mpegts",
            options={"pat_period": "0.02"},  # PAT/PMT every 20ms
        )
        self._stream = self._container.add_stream(
            "aac", rate=sample_rate,
        )
        self._stream.bit_rate = bitrate
        self._stream.layout = "mono"
        self._pts = 0
        self._last_pos = 0

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def segment_durations(self) -> list[float]:
        return list(self._durations)

    def add_segment(self, pcm_bytes: bytes) -> bytes:
        """Encode PCM int16 mono audio through the persistent encoder.

        Returns the new MPEG-TS bytes produced (stored for get_segment()).
        """
        if self._finalized:
            raise RuntimeError("Cannot add segments after finalization")

        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            return b""

        frame = av.AudioFrame(format="s16", layout="mono", samples=n_samples)
        frame.planes[0].update(pcm_bytes)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        self._pts += n_samples

        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        # Extract only the new bytes written since the last segment
        self._buf.seek(0, 2)
        end_pos = self._buf.tell()
        self._buf.seek(self._last_pos)
        seg_data = self._buf.read(end_pos - self._last_pos)
        self._last_pos = end_pos

        duration = n_samples / self._sample_rate
        self._segments.append(seg_data)
        self._durations.append(duration)
        return seg_data

    def get_segment(self, index: int) -> bytes | None:
        """Retrieve a stored segment by index."""
        if 0 <= index < len(self._segments):
            return self._segments[index]
        return None

    def finalize(self) -> None:
        """Flush the encoder and mark the stream as complete."""
        if self._finalized:
            return

        # Flush remaining AAC frames from encoder buffer
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()

        # Append flush bytes to the last segment
        self._buf.seek(0, 2)
        end_pos = self._buf.tell()
        if end_pos > self._last_pos:
            self._buf.seek(self._last_pos)
            remaining = self._buf.read()
            if self._segments:
                self._segments[-1] += remaining
            else:
                self._segments.append(remaining)
                self._durations.append(0)
            self._last_pos = end_pos

        self._finalized = True

    def playlist(self) -> str:
        """Generate m3u8 playlist text for current segments."""
        if not self._durations:
            target_dur = 1
        else:
            target_dur = math.ceil(max(self._durations))

        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_dur}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
            "#EXT-X-START:TIME-OFFSET=0",
        ]

        for i, dur in enumerate(self._durations):
            lines.append(f"#EXTINF:{dur:.6f},")
            lines.append(f"seg{i}.ts")

        if self._finalized:
            lines.append("#EXT-X-ENDLIST")

        return "\n".join(lines) + "\n"

    def close(self) -> None:
        """Release resources. Segments remain accessible."""
        if not self._finalized:
            self.finalize()
        self._buf.close()

    def __del__(self):
        try:
            if hasattr(self, "_container") and not self._finalized:
                self._container.close()
        except Exception:
            pass
