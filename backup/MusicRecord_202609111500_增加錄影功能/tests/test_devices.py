"""Tests for default playback device resolution."""

from types import SimpleNamespace

import pytest

from music_recorder import devices as devices_mod
from music_recorder.devices import get_default_playback_device


class FakePa:
    paWASAPI = 1

    def __init__(self, output_name: str, loopbacks: list[dict]) -> None:
        self._output_name = output_name
        self._loopbacks = loopbacks
        self.terminated = False

    def get_host_api_info_by_type(self, api_type):
        assert api_type == FakePa.paWASAPI
        return {"defaultOutputDevice": 5}

    def get_device_info_by_index(self, index):
        assert index == 5
        return {"name": self._output_name, "index": 5}

    def get_loopback_device_info_generator(self):
        yield from self._loopbacks

    def terminate(self):
        self.terminated = True


def test_get_default_playback_device_matches_loopback(monkeypatch):
    fake = FakePa(
        "Headphones (BT)",
        [
            {
                "index": 9,
                "name": "Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000,
            },
            {
                "index": 10,
                "name": "Headphones (BT) [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000,
            },
        ],
    )
    monkeypatch.setattr(
        devices_mod,
        "get_default_playback_device",
        lambda pa=None: devices_mod._resolve_default(fake, FakePa),
    )
    # Call helper directly to avoid importing real pyaudiowpatch in resolve path
    device = devices_mod._resolve_default(fake, FakePa)
    assert device.name == "Headphones (BT)"
    assert device.loopback_index == 10
    assert device.channels == 2


def test_get_default_playback_device_raises_without_loopback():
    fake = FakePa(
        "Mystery Output",
        [
            {
                "index": 9,
                "name": "Other [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000,
            }
        ],
    )
    with pytest.raises(RuntimeError, match="迴路擷取"):
        devices_mod._resolve_default(fake, FakePa)


def test_get_default_playback_device_raises_without_speaker():
    fake = SimpleNamespace(
        get_host_api_info_by_type=lambda *_: {"defaultOutputDevice": -1},
    )
    with pytest.raises(RuntimeError, match="預設播放裝置"):
        devices_mod._resolve_default(fake, FakePa)
