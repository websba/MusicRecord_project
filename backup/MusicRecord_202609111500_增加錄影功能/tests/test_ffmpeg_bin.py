"""Tests for ffmpeg command helpers (no real encoding)."""

from pathlib import Path

import pytest

from music_recorder.ffmpeg_bin import (
    build_gdigrab_cmd,
    build_mux_cmd,
    video_encoder_args,
)


def test_video_encoder_prefers_h264_mf():
    args = video_encoder_args(["libx264", "h264_mf", "aac"])
    assert args[:2] == ["-c:v", "h264_mf"]
    assert "yuv420p" in args
    assert "quality" in args


def test_video_encoder_falls_back_to_libx264():
    args = video_encoder_args(["libx264", "aac"])
    assert args[:2] == ["-c:v", "libx264"]


def test_video_encoder_raises_without_h264():
    with pytest.raises(RuntimeError, match="H.264"):
        video_encoder_args(["mpeg4", "aac"])


def test_gdigrab_cmd_uses_desktop_and_even_scale(tmp_path: Path):
    out = tmp_path / "v.mp4"
    cmd = build_gdigrab_cmd("ffmpeg", out, ["-c:v", "h264_mf"], fps=30)
    assert cmd[:5] == ["ffmpeg", "-y", "-f", "gdigrab", "-framerate"]
    assert "desktop" in cmd
    assert any("trunc(iw/2)*2" in part for part in cmd)
    assert str(out) == cmd[-1]


def test_mux_cmd_includes_aac_when_audio_present(tmp_path: Path):
    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    out = tmp_path / "o.mp4"
    cmd = build_mux_cmd("ffmpeg", video, out, audio)
    assert "-c:a" in cmd
    assert "aac" in cmd
    assert "+faststart" in cmd


def test_mux_cmd_video_only(tmp_path: Path):
    video = tmp_path / "v.mp4"
    out = tmp_path / "o.mp4"
    cmd = build_mux_cmd("ffmpeg", video, out, None)
    assert "-an" in cmd
    assert "-c:a" not in cmd
