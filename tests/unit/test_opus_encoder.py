"""Tests for OpusEncoder — PCM int16 to OGG/Opus encoding."""

import struct

import numpy as np
import pytest


@pytest.fixture
def encoder():
    from chatterstream.opus_encoder import OpusEncoder
    enc = OpusEncoder()
    yield enc
    enc.close()


def _make_encoder(**kwargs):
    from chatterstream.opus_encoder import OpusEncoder
    return OpusEncoder(**kwargs)


def _make_pcm(n_samples=24000, freq=440, sample_rate=24000):
    """Generate a sine wave as PCM int16 bytes."""
    t = np.linspace(0, 2 * np.pi * freq * n_samples / sample_rate, n_samples)
    samples = (np.sin(t) * 16000).astype(np.int16)
    return samples.tobytes()


class TestOpusEncoderOutput:
    def test_encode_returns_bytes(self, encoder):
        pcm = _make_pcm(24000)
        result = encoder.encode(pcm)
        assert isinstance(result, bytes)

    def test_encode_produces_data(self, encoder):
        """Two calls should produce some OGG data (first may be header only)."""
        chunk1 = encoder.encode(_make_pcm(24000))
        chunk2 = encoder.encode(_make_pcm(24000))
        assert len(chunk1) + len(chunk2) > 0

    def test_ogg_opus_header(self, encoder):
        """First output should contain OGG sync pattern."""
        result = encoder.encode(_make_pcm(24000))
        assert result[:4] == b"OggS"


class TestOpusEncoderFinalize:
    def test_finalize_returns_bytes(self, encoder):
        encoder.encode(_make_pcm(24000))
        result = encoder.finalize()
        assert isinstance(result, bytes)

    def test_finalize_flushes_remaining_data(self, encoder):
        """Finalize should flush buffered Opus frames."""
        encoder.encode(_make_pcm(24000))
        result = encoder.finalize()
        assert len(result) > 0

    def test_full_stream_is_valid_ogg(self, encoder):
        """Header + data + finalize should produce parseable OGG/Opus."""
        import av
        import io

        parts = []
        parts.append(encoder.encode(_make_pcm(24000)))
        parts.append(encoder.encode(_make_pcm(24000)))
        parts.append(encoder.finalize())
        ogg_data = b"".join(parts)

        container = av.open(io.BytesIO(ogg_data), mode="r")
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        container.close()


class TestOpusEncoderReset:
    def test_reset_allows_reuse(self, encoder):
        encoder.encode(_make_pcm(24000))
        encoder.finalize()
        encoder.reset()
        result = encoder.encode(_make_pcm(24000))
        assert isinstance(result, bytes)
        assert result[:4] == b"OggS"  # Fresh OGG header

    def test_reset_produces_independent_streams(self, encoder):
        """Two generation cycles should produce independently valid OGG streams."""
        import av
        import io

        # First generation
        parts1 = [encoder.encode(_make_pcm(24000)), encoder.finalize()]
        ogg1 = b"".join(parts1)

        # Reset and second generation
        encoder.reset()
        parts2 = [encoder.encode(_make_pcm(24000)), encoder.finalize()]
        ogg2 = b"".join(parts2)

        # Both should be independently valid
        for ogg_data in (ogg1, ogg2):
            container = av.open(io.BytesIO(ogg_data), mode="r")
            assert container.streams.audio[0].codec_context.name == "opus"
            container.close()


class TestOpusEncoderConfig:
    def test_bitrate_configurable(self):
        enc = _make_encoder(bitrate=128000)
        pcm = _make_pcm(24000)
        enc.encode(pcm)
        result = enc.finalize()
        enc.close()
        assert len(result) > 0

    def test_default_input_sample_rate_24khz(self, encoder):
        """Default should accept 24kHz PCM (our TTS output rate)."""
        pcm = _make_pcm(24000, sample_rate=24000)
        result = encoder.encode(pcm)
        assert isinstance(result, bytes)


class TestOpusEncoderResampling:
    def test_resamples_to_48khz(self):
        """Opus requires 48kHz; encoder should handle 24kHz input transparently."""
        import av
        import io

        enc = _make_encoder(input_sample_rate=24000)
        parts = [enc.encode(_make_pcm(24000)), enc.finalize()]
        ogg_data = b"".join(parts)

        container = av.open(io.BytesIO(ogg_data), mode="r")
        stream = container.streams.audio[0]
        assert stream.rate == 48000
        container.close()

    def test_mono_output(self):
        """Output should be mono."""
        import av
        import io

        enc = _make_encoder()
        parts = [enc.encode(_make_pcm(24000)), enc.finalize()]
        ogg_data = b"".join(parts)

        container = av.open(io.BytesIO(ogg_data), mode="r")
        stream = container.streams.audio[0]
        assert stream.channels == 1
        container.close()
