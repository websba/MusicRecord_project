"""Tests for UI helper logic and stop/save flow (no real audio devices)."""

import time
from pathlib import Path

import pytest

from music_recorder.app import (
    RecorderApp,
    default_downloads_dir,
    default_filename,
    default_output_dir,
    format_elapsed,
)
from music_recorder.capture import LoopbackRecorder
from music_recorder.devices import PlaybackDevice
from tests.test_capture import DeviceSwitcher, FakePyAudio


def test_format_elapsed():
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(65) == "01:05"
    assert format_elapsed(3661) == "01:01:01"


def test_default_filename_pattern():
    name = default_filename()
    assert name.startswith("recording_")
    assert name.endswith(".mp3")


def test_default_output_dir_is_desktop():
    path = default_output_dir()
    assert path.name == "Desktop"
    assert path.is_absolute()
    assert default_downloads_dir() == path


@pytest.fixture
def fake_stack():
    mod = FakePyAudio()
    switcher = DeviceSwitcher()
    recorder = LoopbackRecorder(
        samplerate=48000,
        channels=2,
        poll_interval=1.0,
        pyaudio_module=mod,
        device_resolver=switcher.resolve,
    )
    return mod, switcher, recorder


def test_app_timer_increases_while_recording(fake_stack, monkeypatch):
    _mod, _switcher, recorder = fake_stack

    monkeypatch.setattr(
        "music_recorder.app.get_default_playback_device",
        lambda: PlaybackDevice("Test Speakers", 1, 11, 48000, 2),
    )

    saved: list[str] = []

    def ask_save(name: str, directory: Path):
        path = directory / name
        saved.append(str(path))
        return str(path)

    app = RecorderApp(
        recorder=recorder,
        ask_save_path=ask_save,
        downloads_dir=Path.cwd() / "_test_downloads",
    )
    app.update()
    app.start_recording()
    app.update()
    time.sleep(0.35)
    app.update()
    app._tick()
    assert ":" in app.timer_label.cget("text")

    app.stop_recording()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        app.update()
        if saved and not app._busy:
            break
        time.sleep(0.05)

    assert saved, "expected save dialog to be used"
    assert Path(saved[0]).exists()
    assert app.screen_record_btn.cget("state") == "normal"
    app.destroy()


class FakeScreen:
    def __init__(self) -> None:
        self.is_recording = False
        self.elapsed_seconds = 1.25
        self.on_segment_done = None

    def start(self) -> str:
        self.is_recording = True
        return "Test Speakers"

    def stop(self):
        self.is_recording = False
        return [Path("screen_20260911_120000.mp4")]


def test_app_screen_buttons_mutex(fake_stack, monkeypatch, tmp_path: Path):
    _mod, _switcher, recorder = fake_stack
    monkeypatch.setattr(
        "music_recorder.app.get_default_playback_device",
        lambda: PlaybackDevice("Test Speakers", 1, 11, 48000, 2),
    )
    screen = FakeScreen()
    app = RecorderApp(
        recorder=recorder,
        output_dir=tmp_path,
        screen_recorder=screen,
    )
    app.update()
    app.start_screen_recording()
    app.update()
    assert screen.is_recording is True
    assert str(app.record_btn.cget("state")) == "disabled"
    assert str(app.stop_btn.cget("state")) == "disabled"
    assert str(app.screen_record_btn.cget("state")) == "disabled"
    assert str(app.screen_stop_btn.cget("state")) == "normal"

    app.stop_screen_recording()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        app.update()
        if not app._busy:
            break
        time.sleep(0.05)

    assert screen.is_recording is False
    assert str(app.record_btn.cget("state")) == "normal"
    assert str(app.screen_record_btn.cget("state")) == "normal"
    app.destroy()
