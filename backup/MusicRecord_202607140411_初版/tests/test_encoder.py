"""Tests for PCM to MP3 encoding."""

from pathlib import Path

import numpy as np
import pytest

from music_recorder.encoder import pcm_to_mp3, save_mp3


def _sine(seconds: float = 0.25, sr: int = 44100, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = 0.2 * np.sin(2 * np.pi * freq * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_pcm_to_mp3_produces_mpeg_frame():
    pcm = _sine()
    data = pcm_to_mp3(pcm, 44100, bitrate=128)
    assert isinstance(data, bytes)
    assert len(data) > 100
    # MP3 frames typically start with 0xFFEx sync word
    assert data[0] == 0xFF
    assert (data[1] & 0xE0) == 0xE0


def test_pcm_to_mp3_rejects_empty():
    with pytest.raises(ValueError, match="沒有可編碼"):
        pcm_to_mp3(np.zeros((0, 2), dtype=np.float32), 48000)


def test_save_mp3_writes_file(tmp_path: Path):
    out = tmp_path / "clip.mp3"
    path = save_mp3(out, _sine(), 44100)
    assert path.exists()
    assert path.stat().st_size > 100
    assert path.read_bytes()[:1] == b"\xff"
