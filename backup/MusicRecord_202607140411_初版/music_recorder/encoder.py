"""Encode float32 PCM audio to MP3 using lameenc."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import lameenc
import numpy as np


def pcm_to_mp3(
    pcm: np.ndarray,
    samplerate: int,
    *,
    bitrate: int = 192,
) -> bytes:
    """Convert float32 PCM in range [-1, 1] to MP3 bytes.

    Args:
        pcm: Array shaped (frames,) or (frames, channels).
        samplerate: Sample rate in Hz.
        bitrate: Target MP3 bitrate in kbps.
    """
    if pcm.size == 0:
        raise ValueError("沒有可編碼的音訊資料。")

    arr = np.asarray(pcm, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"不支援的 PCM 形狀：{arr.shape}")

    channels = int(arr.shape[1])
    if channels not in (1, 2):
        # Downmix / trim to stereo for lameenc
        if channels > 2:
            arr = arr[:, :2]
            channels = 2
        else:
            raise ValueError(f"不支援的聲道數：{channels}")

    clipped = np.clip(arr, -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16)
    interleaved = np.ascontiguousarray(pcm_i16).reshape(-1)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(2)

    mp3_data = encoder.encode(interleaved.tobytes())
    mp3_data += encoder.flush()
    if not mp3_data:
        raise RuntimeError("MP3 編碼結果為空。")
    return bytes(mp3_data)


def save_mp3(
    path: Union[str, Path],
    pcm: np.ndarray,
    samplerate: int,
    *,
    bitrate: int = 192,
) -> Path:
    """Encode PCM to MP3 and write to ``path``. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = pcm_to_mp3(pcm, samplerate, bitrate=bitrate)
    out.write_bytes(data)
    return out
