"""WASAPI loopback capture via PyAudioWPatch with default-device following."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional

import numpy as np

from music_recorder.devices import PlaybackDevice, get_default_playback_device

DEFAULT_SAMPLERATE = 48000
DEFAULT_CHANNELS = 2
DEVICE_POLL_INTERVAL_SEC = 1.0
FRAMES_PER_BUFFER = 2048


class LoopbackRecorder:
    """Capture system audio from the current default playback device.

    All PortAudio/WASAPI access stays on one dedicated capture thread.
    """

    def __init__(
        self,
        samplerate: int = DEFAULT_SAMPLERATE,
        channels: int = DEFAULT_CHANNELS,
        poll_interval: float = DEVICE_POLL_INTERVAL_SEC,
        pyaudio_module: Any = None,
        device_resolver: Optional[Callable[[Any], PlaybackDevice]] = None,
        keep_chunks: bool = True,
        chunk_sink: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.poll_interval = poll_interval
        self.keep_chunks = keep_chunks
        self.chunk_sink = chunk_sink
        self._pyaudio_module = pyaudio_module
        self._device_resolver = device_resolver

        self._lock = threading.Lock()
        self._chunks: List[np.ndarray] = []
        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._device_name: str = ""
        self._error: Optional[BaseException] = None
        self._started_at: Optional[float] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None

    def _resolve(self, pa: Any) -> PlaybackDevice:
        if self._device_resolver is not None:
            return self._device_resolver(pa)
        return get_default_playback_device(pa)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def device_name(self) -> str:
        with self._lock:
            return self._device_name

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def start(self) -> str:
        """Start loopback recording. Returns the current device name."""
        if self._recording or self._thread is not None:
            raise RuntimeError("已在錄音中。")

        with self._lock:
            self._chunks = []
            self._device_name = ""
            self._error = None
            self._start_error = None
            self._started_at = time.monotonic()
            self._recording = True

        self._ready.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="loopback-capture",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=8.0):
            self._recording = False
            self._thread.join(timeout=2.0)
            self._thread = None
            raise RuntimeError("無法在時限內啟動錄音。")

        if self._start_error is not None:
            err = self._start_error
            self._start_error = None
            self._thread.join(timeout=2.0)
            self._thread = None
            self._recording = False
            raise RuntimeError(f"無法開始錄音：{err}") from err

        return self.device_name

    def stop(self) -> np.ndarray:
        """Stop recording and return concatenated float32 PCM frames."""
        if not self._recording and self._thread is None:
            return np.zeros((0, self.channels), dtype=np.float32)

        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        if self._error is not None:
            err = self._error
            self._error = None
            raise RuntimeError(f"錄音失敗：{err}") from err

        with self._lock:
            chunks = list(self._chunks)
            self._chunks = []
            self._started_at = None

        if not chunks:
            return np.zeros((0, self.channels), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _open_stream(self, pa: Any, pyaudio_mod: Any, device: PlaybackDevice):
        channels = min(self.channels, device.channels) or 1
        rate = device.samplerate or self.samplerate
        stream = pa.open(
            format=pyaudio_mod.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device.loopback_index,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        with self._lock:
            self._device_name = device.name
            self.channels = channels
            self.samplerate = rate
        return stream

    def _close_stream(self, stream: Any) -> None:
        if stream is None:
            return
        try:
            if stream.is_active():
                stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _capture_loop(self) -> None:
        mod = self._pyaudio_module
        if mod is None:
            import pyaudiowpatch as mod  # type: ignore

        pa = None
        stream = None
        current_key: Optional[int] = None
        next_device_check = 0.0
        ready_set = False

        try:
            pa = mod.PyAudio()
            while self._recording:
                now = time.monotonic()
                if stream is None or now >= next_device_check:
                    next_device_check = now + self.poll_interval
                    device = self._resolve(pa)
                    if device.loopback_index != current_key:
                        self._close_stream(stream)
                        stream = None
                        time.sleep(0.05)
                        stream = self._open_stream(pa, mod, device)
                        current_key = device.loopback_index

                    if not ready_set:
                        ready_set = True
                        self._ready.set()

                assert stream is not None
                raw = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                arr = np.frombuffer(raw, dtype=np.float32)
                if arr.size == 0:
                    continue
                channels = self.channels
                frames = arr.size // channels
                if frames <= 0:
                    continue
                arr = arr[: frames * channels].reshape(frames, channels).copy()
                sink = self.chunk_sink
                with self._lock:
                    if self.keep_chunks:
                        self._chunks.append(arr)
                if sink is not None:
                    try:
                        sink(arr)
                    except Exception as exc:  # noqa: BLE001
                        self._error = exc
                        self._recording = False
                        break
        except BaseException as exc:  # noqa: BLE001
            if not ready_set:
                self._start_error = exc
                self._ready.set()
            else:
                self._error = exc
            self._recording = False
        finally:
            self._close_stream(stream)
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
            if not ready_set:
                self._ready.set()
