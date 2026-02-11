"""Tests for HLSSegmenter — PCM int16 to MPEG-TS/AAC HLS segments."""

import io

import numpy as np
import pytest

av = pytest.importorskip("av", reason="PyAV required for media tests")


def _make_segmenter(**kwargs):
    from chatterstream.hls_segmenter import HLSSegmenter
    return HLSSegmenter(**kwargs)


def _make_pcm(n_samples=24000, freq=440, sample_rate=24000):
    """Generate a sine wave as PCM int16 bytes."""
    t = np.linspace(0, 2 * np.pi * freq * n_samples / sample_rate, n_samples)
    samples = (np.sin(t) * 16000).astype(np.int16)
    return samples.tobytes()


class TestHLSSegmenterOutput:
    def test_add_segment_returns_bytes(self):
        seg = _make_segmenter()
        result = seg.add_segment(_make_pcm(24000))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_segment_starts_with_ts_sync(self):
        """MPEG-TS packets start with 0x47 sync byte."""
        seg = _make_segmenter()
        result = seg.add_segment(_make_pcm(24000))
        assert result[0] == 0x47

    def test_segment_is_valid_mpegts(self):
        """Segment should be parseable as MPEG-TS with AAC audio."""
        seg = _make_segmenter()
        ts_data = seg.add_segment(_make_pcm(24000))
        container = av.open(io.BytesIO(ts_data), mode="r")
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac"
        container.close()

    def test_segment_mono_output(self):
        seg = _make_segmenter()
        ts_data = seg.add_segment(_make_pcm(24000))
        container = av.open(io.BytesIO(ts_data), mode="r")
        assert container.streams.audio[0].channels == 1
        container.close()

    def test_segment_sample_rate(self):
        seg = _make_segmenter(sample_rate=24000)
        ts_data = seg.add_segment(_make_pcm(24000))
        container = av.open(io.BytesIO(ts_data), mode="r")
        assert container.streams.audio[0].rate == 24000
        container.close()


class TestHLSSegmenterMultiple:
    def test_multiple_segments_independent(self):
        """Each segment is independently parseable."""
        seg = _make_segmenter()
        seg1 = seg.add_segment(_make_pcm(24000))
        seg2 = seg.add_segment(_make_pcm(24000))

        for ts_data in (seg1, seg2):
            container = av.open(io.BytesIO(ts_data), mode="r")
            assert container.streams.audio[0].codec_context.name == "aac"
            container.close()

    def test_segment_count_tracked(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))
        assert seg.segment_count == 3

    def test_segment_durations_tracked(self):
        seg = _make_segmenter(sample_rate=24000)
        seg.add_segment(_make_pcm(24000))  # 1 second
        seg.add_segment(_make_pcm(12000))  # 0.5 seconds
        durations = seg.segment_durations
        assert len(durations) == 2
        assert abs(durations[0] - 1.0) < 0.01
        assert abs(durations[1] - 0.5) < 0.01


class TestHLSSegmenterPlaylist:
    def test_playlist_has_header(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        assert "#EXTM3U" in playlist
        assert "#EXT-X-VERSION:" in playlist
        assert "#EXT-X-TARGETDURATION:" in playlist

    def test_playlist_has_segments(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        assert "seg0.ts" in playlist
        assert "seg1.ts" in playlist
        assert "#EXTINF:" in playlist

    def test_playlist_no_endlist_during_streaming(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        assert "#EXT-X-ENDLIST" not in playlist

    def test_playlist_has_endlist_when_finalized(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.finalize()
        playlist = seg.playlist()
        assert "#EXT-X-ENDLIST" in playlist

    def test_playlist_event_type(self):
        """EVENT type means segments are only appended, never removed."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        assert "#EXT-X-PLAYLIST-TYPE:EVENT" in playlist

    def test_target_duration_covers_longest_segment(self):
        seg = _make_segmenter(sample_rate=24000)
        seg.add_segment(_make_pcm(24000))   # 1s
        seg.add_segment(_make_pcm(48000))   # 2s
        playlist = seg.playlist()
        # Target duration should be >= ceil of longest segment
        assert "#EXT-X-TARGETDURATION:2" in playlist


class TestHLSSegmenterConfig:
    def test_bitrate_configurable(self):
        seg = _make_segmenter(bitrate=128000)
        result = seg.add_segment(_make_pcm(24000))
        assert len(result) > 0

    def test_empty_pcm_returns_empty(self):
        seg = _make_segmenter()
        result = seg.add_segment(b"")
        assert result == b""


class TestHLSSegmenterStartOffset:
    def test_playlist_has_start_offset_zero(self):
        """EXT-X-START tells Safari to begin playback immediately."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        assert "#EXT-X-START:TIME-OFFSET=0" in playlist

    def test_start_offset_before_segments(self):
        """START tag appears in header section before segment entries."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        playlist = seg.playlist()
        start_pos = playlist.index("#EXT-X-START:")
        first_extinf = playlist.index("#EXTINF:")
        assert start_pos < first_extinf


class TestHLSSegmenterContinuousEncoder:
    """Persistent encoder eliminates MDCT boundary artifacts."""

    def test_concatenated_segments_form_valid_stream(self):
        """All segments concatenated should be a valid MPEG-TS stream."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))
        seg.finalize()

        full_stream = b"".join(seg.get_segment(i) for i in range(seg.segment_count))
        container = av.open(io.BytesIO(full_stream), mode="r")
        total_samples = sum(f.samples for f in container.decode(audio=0))
        container.close()

        # 3 segments of 1s each at 24kHz ≈ 72000 samples
        # AAC frame size varies, allow tolerance
        assert total_samples > 60000
        assert total_samples < 84000

    def test_second_segment_parseable(self):
        """Segment after the first is independently parseable (has PAT/PMT)."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.add_segment(_make_pcm(24000))

        # Second segment should be parseable on its own
        container = av.open(io.BytesIO(seg.get_segment(1)), mode="r")
        assert container.streams.audio[0].codec_context.name == "aac"
        container.close()

    def test_finalize_produces_valid_complete_stream(self):
        """After finalize, full stream decodes to expected duration."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))  # 1s
        seg.finalize()

        full_stream = b"".join(seg.get_segment(i) for i in range(seg.segment_count))
        container = av.open(io.BytesIO(full_stream), mode="r")
        total_samples = sum(f.samples for f in container.decode(audio=0))
        container.close()
        # 1s at 24kHz = 24000, AAC adds/removes a few due to frame alignment
        assert total_samples > 20000
        assert total_samples < 28000

    def test_add_after_finalize_raises(self):
        """Cannot add segments after finalization."""
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.finalize()
        with pytest.raises(RuntimeError):
            seg.add_segment(_make_pcm(24000))

    def test_silence_primer_no_artifact(self):
        """Silence followed by audio through same encoder produces clean stream."""
        seg = _make_segmenter()
        # Silence primer
        silence = b"\x00" * (12000 * 2)  # 0.5s silence
        seg.add_segment(silence)
        # Real audio
        seg.add_segment(_make_pcm(24000))
        seg.finalize()

        full_stream = b"".join(seg.get_segment(i) for i in range(seg.segment_count))
        container = av.open(io.BytesIO(full_stream), mode="r")
        total_samples = sum(f.samples for f in container.decode(audio=0))
        container.close()
        # 0.5s silence + 1s audio at 24kHz = ~36000 samples
        assert total_samples > 28000


class TestHLSSegmenterGetSegment:
    def test_get_segment_by_index(self):
        seg = _make_segmenter()
        data0 = seg.add_segment(_make_pcm(24000))
        data1 = seg.add_segment(_make_pcm(24000))
        assert seg.get_segment(0) == data0
        assert seg.get_segment(1) == data1

    def test_get_segment_out_of_range(self):
        seg = _make_segmenter()
        assert seg.get_segment(0) is None

    def test_close_releases_resources(self):
        seg = _make_segmenter()
        seg.add_segment(_make_pcm(24000))
        seg.close()
        # After close, segments are still accessible
        assert seg.get_segment(0) is not None
