import pytest

from voice_arm.audio import chunk_pcm
from voice_arm.config import FRAME_SIZE_BYTES, INPUT_SAMPLE_RATE_HZ


def test_frame_size_matches_20ms_at_16khz():
    # 20 ms * 16 kHz * 2 bytes (s16le) * 1 channel = 640 bytes
    assert FRAME_SIZE_BYTES == INPUT_SAMPLE_RATE_HZ * 20 // 1000 * 2


def test_chunk_pcm_splits_into_fixed_frames():
    buf = b"\x00" * (FRAME_SIZE_BYTES * 3)
    frames = list(chunk_pcm(buf))
    assert len(frames) == 3
    assert all(len(f) == FRAME_SIZE_BYTES for f in frames)


def test_chunk_pcm_drops_trailing_partial_frame():
    buf = b"\x00" * (FRAME_SIZE_BYTES * 2 + 17)
    frames = list(chunk_pcm(buf))
    assert len(frames) == 2


def test_chunk_pcm_rejects_bad_frame_size():
    with pytest.raises(ValueError):
        list(chunk_pcm(b"abc", frame_size=0))


def test_chunk_pcm_empty_input_yields_nothing():
    assert list(chunk_pcm(b"")) == []
