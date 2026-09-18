"""Locate ffmpeg and build capture / mux command lines."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

GRAB_FPS = 30
SCALE_EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def popen_kwargs() -> dict:
    """Flags so ffmpeg does not flash a console window on Windows."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("未安裝 imageio-ffmpeg，無法進行螢幕錄影。") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg_help(ffmpeg: str, args: Sequence[str]) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        **popen_kwargs(),
    )
    return (result.stdout or "") + (result.stderr or "")


@lru_cache(maxsize=4)
def list_encoders(ffmpeg: Optional[str] = None) -> frozenset[str]:
    exe = ffmpeg or get_ffmpeg_exe()
    text = _run_ffmpeg_help(exe, ["-encoders"])
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) >= 2 and len(parts[0]) >= 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


@lru_cache(maxsize=4)
def has_gdigrab(ffmpeg: Optional[str] = None) -> bool:
    exe = ffmpeg or get_ffmpeg_exe()
    text = _run_ffmpeg_help(exe, ["-formats"])
    return "gdigrab" in text


def video_encoder_args(encoders: Optional[Iterable[str]] = None) -> List[str]:
    available = set(encoders) if encoders is not None else set(list_encoders())
    if "h264_mf" in available:
        return [
            "-c:v",
            "h264_mf",
            "-scenario",
            "archive",
            "-rate_control",
            "quality",
            "-quality",
            "70",
            "-pix_fmt",
            "yuv420p",
        ]
    if "libx264" in available:
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    raise RuntimeError("找不到可用的 H.264 編碼器（h264_mf / libx264）。")


def build_gdigrab_cmd(
    ffmpeg: str,
    output: Path,
    encoder_args: Sequence[str],
    *,
    fps: int = GRAB_FPS,
) -> List[str]:
    return [
        ffmpeg,
        "-y",
        "-f",
        "gdigrab",
        "-framerate",
        str(fps),
        "-draw_mouse",
        "1",
        "-i",
        "desktop",
        "-vf",
        SCALE_EVEN,
        *encoder_args,
        str(output),
    ]


def build_rawvideo_cmd(
    ffmpeg: str,
    output: Path,
    encoder_args: Sequence[str],
    *,
    width: int,
    height: int,
    fps: int = GRAB_FPS,
    pix_fmt: str = "bgra",
) -> List[str]:
    return [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        SCALE_EVEN,
        *encoder_args,
        str(output),
    ]


def build_mux_cmd(
    ffmpeg: str,
    video: Path,
    output: Path,
    audio: Optional[Path] = None,
) -> List[str]:
    cmd = [ffmpeg, "-y", "-i", str(video)]
    if audio is not None:
        cmd += [
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]
    else:
        cmd += ["-c:v", "copy", "-an"]
    cmd += ["-movflags", "+faststart", str(output)]
    return cmd
