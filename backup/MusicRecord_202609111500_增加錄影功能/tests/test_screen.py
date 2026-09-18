"""Tests for screen recording rotation and mux (fake ffmpeg / devices)."""

import time
from datetime import datetime, timedelta
from pathlib import Path

from music_recorder.capture import LoopbackRecorder
from music_recorder.screen import (
    ScreenRecorder,
    default_screen_filename,
    unique_output_path,
)
from tests.test_capture import DeviceSwitcher, FakePyAudio


class FakeGrab:
    def __init__(self, dest: Path) -> None:
        dest.write_bytes(b"fake-video")
        self.stopped = False

    def request_stop(self) -> None:
        self.stopped = True

    def wait(self, timeout: float) -> None:
        return None


class FakeBackend:
    def __init__(self) -> None:
        self.grabs = 0
        self.mux_calls: list[tuple[Path, Path | None, Path]] = []

    def start_grab(self, dest: Path) -> FakeGrab:
        self.grabs += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        return FakeGrab(dest)

    def mux(self, video: Path, audio: Path | None, dest: Path) -> None:
        self.mux_calls.append((video, audio, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = b"mp4"
        if audio is not None and audio.exists():
            payload += b"-a"
        dest.write_bytes(payload)


class SeqClock:
    def __init__(self) -> None:
        self.i = 0

    def __call__(self) -> datetime:
        stamp = datetime(2026, 9, 11, 12, 0, 0) + timedelta(hours=self.i)
        self.i += 1
        return stamp


def _recorder() -> LoopbackRecorder:
    return LoopbackRecorder(
        samplerate=48000,
        channels=2,
        poll_interval=1.0,
        pyaudio_module=FakePyAudio(),
        device_resolver=DeviceSwitcher().resolve,
    )


def test_default_screen_filename():
    name = default_screen_filename(datetime(2026, 9, 11, 14, 30, 5))
    assert name == "screen_20260911_143005.mp4"


def test_unique_output_path_adds_suffix(tmp_path: Path):
    first = unique_output_path(tmp_path, "screen_a.mp4")
    first.write_bytes(b"x")
    second = unique_output_path(tmp_path, "screen_a.mp4")
    assert second != first
    assert second.name == "screen_a_2.mp4"


def test_screen_recorder_stop_saves_last_segment(tmp_path: Path):
    backend = FakeBackend()
    out_dir = tmp_path / "desktop"
    rec = ScreenRecorder(
        recorder=_recorder(),
        output_dir=out_dir,
        segment_seconds=3600,
        backend=backend,
        clock=SeqClock(),
        temp_dir=tmp_path / "tmp",
    )
    rec.start()
    time.sleep(0.25)
    paths = rec.stop()
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].parent == out_dir
    assert paths[0].name.startswith("screen_")
    assert backend.grabs == 1
    assert len(backend.mux_calls) == 1
    assert rec.is_recording is False


def test_screen_recorder_splits_hourly(tmp_path: Path):
    backend = FakeBackend()
    rec = ScreenRecorder(
        recorder=_recorder(),
        output_dir=tmp_path / "desktop",
        segment_seconds=0.35,
        backend=backend,
        clock=SeqClock(),
        temp_dir=tmp_path / "tmp",
    )
    rec.start()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if len(rec.saved_paths) >= 1 and backend.grabs >= 2:
            break
        time.sleep(0.05)
    paths = rec.stop()
    assert len(paths) >= 2
    assert all(p.exists() for p in paths)
    assert backend.grabs >= 2
    names = [p.name for p in paths]
    assert len(names) == len(set(names))
