"""Tests for loopback capture and device following."""

import time
from types import SimpleNamespace

import numpy as np

from music_recorder.capture import LoopbackRecorder
from music_recorder.devices import PlaybackDevice


class FakeStream:
    def __init__(self) -> None:
        self.closed = False
        self._n = 0

    def is_active(self):
        return not self.closed

    def stop_stream(self):
        pass

    def close(self):
        self.closed = True

    def read(self, frames, exception_on_overflow=False):
        self._n += 1
        # stereo float32 silence-ish tone
        data = np.full((frames, 2), 0.1, dtype=np.float32)
        return data.tobytes()


class FakePyAudio:
    paFloat32 = 1
    paWASAPI = 1

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.terminated = False

    def PyAudio(self):
        return self

    def open(self, **kwargs):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.terminated = True


class DeviceSwitcher:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, pa) -> PlaybackDevice:
        self.calls += 1
        if self.calls <= 2:
            return PlaybackDevice("Speakers", 1, 11, 48000, 2)
        return PlaybackDevice("Bluetooth Headset", 2, 22, 48000, 2)


def test_recorder_start_stop_returns_pcm():
    mod = FakePyAudio()
    switcher = DeviceSwitcher()
    rec = LoopbackRecorder(
        samplerate=48000,
        channels=2,
        poll_interval=0.05,
        pyaudio_module=mod,
        device_resolver=switcher.resolve,
    )
    name = rec.start()
    assert name == "Speakers"
    time.sleep(0.3)
    pcm = rec.stop()
    assert pcm.ndim == 2
    assert pcm.shape[1] == 2
    assert pcm.shape[0] > 0
    assert rec.is_recording is False
    assert mod.terminated is True


def test_recorder_follows_device_change():
    mod = FakePyAudio()
    switcher = DeviceSwitcher()
    rec = LoopbackRecorder(
        samplerate=48000,
        channels=2,
        poll_interval=0.05,
        pyaudio_module=mod,
        device_resolver=switcher.resolve,
    )
    rec.start()
    deadline = time.time() + 2.0
    seen_bt = False
    while time.time() < deadline:
        if rec.device_name == "Bluetooth Headset":
            seen_bt = True
            break
        time.sleep(0.05)
    pcm = rec.stop()
    assert seen_bt
    assert len(mod.streams) >= 2
    assert mod.streams[0].closed is True
    assert pcm.shape[0] > 0
