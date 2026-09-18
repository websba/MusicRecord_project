"""Query the current default playback device and its WASAPI loopback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PlaybackDevice:
    """Default output device and matching loopback endpoint."""

    name: str
    output_index: int
    loopback_index: int
    samplerate: int
    channels: int


def get_default_playback_device(pa: Any = None) -> PlaybackDevice:
    """Return loopback info for the current default WASAPI output device.

    Args:
        pa: Optional open ``PyAudio`` instance. If omitted, a temporary
            instance is created and terminated before returning.
    """
    import pyaudiowpatch as pyaudio  # type: ignore

    owns_instance = False
    if pa is None:
        pa = pyaudio.PyAudio()
        owns_instance = True
    try:
        return _resolve_default(pa, pyaudio)
    finally:
        if owns_instance:
            pa.terminate()


def _resolve_default(pa: Any, pyaudio_mod: Any) -> PlaybackDevice:
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
    except Exception as exc:
        raise RuntimeError("此系統沒有可用的 WASAPI 音訊介面。") from exc

    out_index = wasapi.get("defaultOutputDevice")
    if out_index is None or int(out_index) < 0:
        raise RuntimeError("找不到預設播放裝置。")

    output = pa.get_device_info_by_index(int(out_index))
    loopback = _find_loopback(pa, output["name"])
    if loopback is None:
        raise RuntimeError(f"找不到播放裝置的迴路擷取來源：{output['name']}")

    channels = max(1, min(2, int(loopback.get("maxInputChannels") or 2)))
    samplerate = int(loopback.get("defaultSampleRate") or 48000)
    return PlaybackDevice(
        name=output["name"],
        output_index=int(out_index),
        loopback_index=int(loopback["index"]),
        samplerate=samplerate,
        channels=channels,
    )


def _find_loopback(pa: Any, output_name: str) -> Optional[dict]:
    candidates = list(pa.get_loopback_device_info_generator())
    for device in candidates:
        name = device.get("name") or ""
        bare = name.replace(" [Loopback]", "")
        if output_name == bare or output_name in name:
            return device
    for device in candidates:
        name = device.get("name") or ""
        bare = name.replace(" [Loopback]", "")
        if bare in output_name:
            return device
    return None
