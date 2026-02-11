"""Tests for AudioPostProcessor — tensor to PCM int16 bytes with crossfade."""

import struct

import torch
import pytest


def _make_processor():
    from chatterstream.audio_post_processor import AudioPostProcessor
    return AudioPostProcessor(sample_rate=24000)


# At 24kHz with 20ms crossfade, the tail reserve is 480 samples.
_CROSSFADE_SAMPLES = 480


class TestAudioPostProcessorOutput:
    def test_returns_bytes(self):
        proc = _make_processor()
        audio = torch.randn(1, 4800)  # 200ms at 24kHz
        result = proc.process(audio, chunk_index=0)
        assert isinstance(result, bytes)

    def test_int16_format(self):
        proc = _make_processor()
        # Use large enough audio that tail trim still leaves samples
        audio = torch.zeros(1, 2400)
        result = proc.process(audio, chunk_index=0)
        # Output = input - crossfade tail (each sample is 2 bytes)
        expected_samples = 2400 - _CROSSFADE_SAMPLES
        assert len(result) == expected_samples * 2

    def test_byte_length_accounts_for_crossfade_tail(self):
        proc = _make_processor()
        n_samples = 4800
        audio = torch.randn(1, n_samples)
        result = proc.process(audio, chunk_index=0)
        # Tail is held back for crossfade with next chunk
        expected_samples = n_samples - _CROSSFADE_SAMPLES
        assert len(result) == expected_samples * 2


class TestAudioPostProcessorFadeIn:
    def test_fade_in_on_chunk_zero(self):
        proc = _make_processor()
        audio = torch.ones(1, 4800)
        result = proc.process(audio, chunk_index=0)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        # First sample should be near-zero (silence portion of fade)
        assert abs(samples[0]) < 100

    def test_no_fade_on_subsequent_chunks(self):
        proc = _make_processor()
        # Process chunk 0 first to establish prev_tail
        proc.process(torch.ones(1, 4800), chunk_index=0)
        # Chunk 1 gets crossfade applied but no fade-in
        audio = torch.ones(1, 4800)
        result = proc.process(audio, chunk_index=1)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        # After crossfade zone, samples should be near max
        assert abs(samples[_CROSSFADE_SAMPLES]) > 30000

    def test_fade_in_only_affects_start(self):
        proc = _make_processor()
        audio = torch.ones(1, 4800)
        result = proc.process(audio, chunk_index=0)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        # End of output (before tail was trimmed) should be unaffected
        assert abs(samples[-1]) > 30000


class TestAudioPostProcessorClipping:
    def test_clips_to_int16_range(self):
        proc = _make_processor()
        audio = torch.full((1, 2400), 2.0)
        result = proc.process(audio, chunk_index=1)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        assert max(samples) <= 32767
        assert min(samples) >= -32768


class TestAudioPostProcessorSqueeze:
    def test_handles_2d_input(self):
        proc = _make_processor()
        audio = torch.randn(1, 2400)
        result = proc.process(audio, chunk_index=0)
        assert isinstance(result, bytes)

    def test_handles_1d_input(self):
        proc = _make_processor()
        audio = torch.randn(2400)
        result = proc.process(audio, chunk_index=0)
        assert isinstance(result, bytes)
        expected_samples = 2400 - _CROSSFADE_SAMPLES
        assert len(result) == expected_samples * 2


class TestAudioPostProcessorDevice:
    def test_cpu_transfer(self):
        proc = _make_processor()
        audio = torch.randn(1, 2400)
        result = proc.process(audio, chunk_index=0)
        assert isinstance(result, bytes)


class TestAudioPostProcessorCrossfade:
    def test_crossfade_blends_chunks(self):
        """Chunk boundary should be blended, not a hard cut."""
        proc = _make_processor()
        # Chunk 0: all ones
        proc.process(torch.ones(1, 4800), chunk_index=0)
        # Chunk 1: all negative ones — boundary should be blended
        result = proc.process(-torch.ones(1, 4800), chunk_index=1)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        # Middle of crossfade zone should be near zero (blend of +1 and -1)
        mid = _CROSSFADE_SAMPLES // 2
        assert abs(samples[mid]) < 5000  # blended, not hard cut

    def test_reset_clears_crossfade_state(self):
        proc = _make_processor()
        proc.process(torch.ones(1, 4800), chunk_index=0)
        proc.reset()
        # After reset, chunk 0 should have no crossfade from previous
        result = proc.process(torch.ones(1, 4800), chunk_index=0)
        n_out = len(result) // 2
        samples = struct.unpack(f"<{n_out}h", result)
        # No crossfade applied, just fade-in
        assert abs(samples[0]) < 100  # fade-in silence

    def test_final_chunk_includes_full_tail(self):
        """When audio is too short for tail reserve, output everything."""
        proc = _make_processor()
        # Very short audio — shorter than crossfade length
        audio = torch.ones(1, 100)
        result = proc.process(audio, chunk_index=0)
        # Should output all samples (no tail reserve for tiny chunks)
        assert len(result) == 100 * 2


class TestAudioPostProcessorFinalize:
    def test_finalize_returns_remaining_tail(self):
        """After processing chunks, finalize() returns the buffered tail."""
        proc = _make_processor()
        proc.process(torch.ones(1, 4800), chunk_index=0)
        tail = proc.finalize()
        assert isinstance(tail, bytes)
        # Tail should be crossfade_len samples * 2 bytes each
        assert len(tail) == _CROSSFADE_SAMPLES * 2

    def test_finalize_on_fresh_processor_returns_empty(self):
        """No tail buffered = empty bytes."""
        proc = _make_processor()
        tail = proc.finalize()
        assert tail == b""

    def test_finalize_clears_state(self):
        """After finalize, a second call returns empty."""
        proc = _make_processor()
        proc.process(torch.ones(1, 4800), chunk_index=0)
        proc.finalize()
        assert proc.finalize() == b""

    def test_finalize_tail_values_are_valid_int16(self):
        """Finalized tail should be clamped int16."""
        proc = _make_processor()
        proc.process(torch.full((1, 4800), 2.0), chunk_index=0)
        tail = proc.finalize()
        n = len(tail) // 2
        samples = struct.unpack(f"<{n}h", tail)
        assert max(samples) <= 32767
        assert min(samples) >= -32768
