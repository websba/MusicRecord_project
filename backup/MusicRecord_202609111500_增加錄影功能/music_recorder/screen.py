"""Screen recording: virtual desktop video + loopback audio, hourly MP4 files."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from music_recorder.capture import LoopbackRecorder
from music_recorder.encoder import StreamingWavWriter
from music_recorder.ffmpeg_bin import (
    GRAB_FPS,
    build_gdigrab_cmd,
    build_mux_cmd,
    build_rawvideo_cmd,
    get_ffmpeg_exe,
    has_gdigrab,
    list_encoders,
    popen_kwargs,
    video_encoder_args,
)

SEGMENT_SECONDS = 3600.0


def default_screen_filename(now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"screen_{stamp}.mp4"


def unique_output_path(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


class GrabSession(Protocol):
    def request_stop(self) -> None: ...

    def wait(self, timeout: float) -> None: ...


class SubprocessGrab:
    """FFmpeg child process writing a video file."""

    def __init__(
        self,
        proc: subprocess.Popen,
        *,
        stop_mode: str,
        writer_stop: Optional[threading.Event] = None,
        writer_thread: Optional[threading.Thread] = None,
    ) -> None:
        self.proc = proc
        self.stop_mode = stop_mode
        self._writer_stop = writer_stop
        self._writer_thread = writer_thread
        self._stderr: List[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="ffmpeg-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stream = self.proc.stderr
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                self._stderr.append(chunk)
                if len(self._stderr) > 24:
                    self._stderr.pop(0)
        except Exception:
            pass

    def stderr_tail(self) -> str:
        try:
            return b"".join(self._stderr).decode("utf-8", errors="replace")[-800:]
        except Exception:
            return ""

    def request_stop(self) -> None:
        if self._writer_stop is not None:
            self._writer_stop.set()
        try:
            if self.proc.stdin is None:
                return
            if self.stop_mode == "q":
                self.proc.stdin.write(b"q\n")
                self.proc.stdin.flush()
            else:
                self.proc.stdin.close()
        except Exception:
            pass

    def wait(self, timeout: float) -> None:
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3)
        self._stderr_thread.join(timeout=1)


class FFmpegBackend:
    """Start desktop capture and mux A/V with a real ffmpeg binary."""

    def __init__(self, ffmpeg: Optional[str] = None) -> None:
        self.ffmpeg = ffmpeg or get_ffmpeg_exe()
        self.encoder_args = video_encoder_args(list_encoders(self.ffmpeg))
        self.use_gdigrab = has_gdigrab(self.ffmpeg)

    def start_grab(self, dest: Path) -> GrabSession:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.use_gdigrab:
            return self._start_gdigrab(dest)
        return self._start_mss(dest)

    def mux(self, video: Path, audio: Optional[Path], dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio_path = audio if audio is not None and audio.exists() else None
        cmd = build_mux_cmd(self.ffmpeg, video, dest, audio_path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            **popen_kwargs(),
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()[-800:]
            raise RuntimeError(f"合成 MP4 失敗：{err or result.returncode}")
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("合成 MP4 結果為空。")

    def _popen(self, cmd: List[str]) -> subprocess.Popen:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            **popen_kwargs(),
        )

    def _start_gdigrab(self, dest: Path) -> SubprocessGrab:
        cmd = build_gdigrab_cmd(self.ffmpeg, dest, self.encoder_args, fps=GRAB_FPS)
        proc = self._popen(cmd)
        session = SubprocessGrab(proc, stop_mode="q")
        time.sleep(0.6)
        if proc.poll() is not None:
            raise RuntimeError(f"無法開始螢幕擷取：{session.stderr_tail() or 'ffmpeg 立即結束'}")
        return session

    def _start_mss(self, dest: Path) -> SubprocessGrab:
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("此 ffmpeg 不支援 gdigrab，且未安裝 mss，無法擷取螢幕。") from exc

        with mss.mss() as probe:
            mon = probe.monitors[0]
            width = int(mon["width"])
            height = int(mon["height"])
        if width <= 0 or height <= 0:
            raise RuntimeError("找不到可用的顯示器。")

        cmd = build_rawvideo_cmd(
            self.ffmpeg,
            dest,
            self.encoder_args,
            width=width,
            height=height,
            fps=GRAB_FPS,
            pix_fmt="bgra",
        )
        proc = self._popen(cmd)
        writer_stop = threading.Event()

        def writer() -> None:
            interval = 1.0 / GRAB_FPS
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[0]
                    next_t = time.monotonic()
                    while not writer_stop.is_set() and proc.poll() is None:
                        frame = sct.grab(monitor)
                        try:
                            if proc.stdin is None:
                                break
                            proc.stdin.write(frame.bgra)
                        except (BrokenPipeError, OSError):
                            break
                        next_t += interval
                        delay = next_t - time.monotonic()
                        if delay > 0:
                            writer_stop.wait(timeout=delay)
            except Exception:
                writer_stop.set()

        thread = threading.Thread(target=writer, name="mss-grab", daemon=True)
        thread.start()
        session = SubprocessGrab(
            proc,
            stop_mode="pipe",
            writer_stop=writer_stop,
            writer_thread=thread,
        )
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(f"無法開始螢幕擷取：{session.stderr_tail() or 'ffmpeg 立即結束'}")
        return session


class ScreenRecorder:
    """Record all displays plus system audio, saving one MP4 per hour."""

    def __init__(
        self,
        recorder: LoopbackRecorder,
        output_dir: Path,
        *,
        segment_seconds: float = SEGMENT_SECONDS,
        backend: Optional[FFmpegBackend] = None,
        clock: Optional[Callable[[], datetime]] = None,
        temp_dir: Optional[Path] = None,
        on_segment_done: Optional[
            Callable[[Optional[Path], Optional[BaseException]], None]
        ] = None,
    ) -> None:
        self._recorder = recorder
        self._output_dir = Path(output_dir)
        self._segment_seconds = float(segment_seconds)
        self._backend = backend
        self._clock = clock or datetime.now
        self._temp_dir_arg = Path(temp_dir) if temp_dir is not None else None
        self.on_segment_done = on_segment_done

        self._recording = False
        self._stop_event = threading.Event()
        self._io_lock = threading.Lock()
        self._rotate_lock = threading.Lock()
        self._segment_index = 0
        self._grab: Optional[GrabSession] = None
        self._wav: Optional[StreamingWavWriter] = None
        self._video_path: Optional[Path] = None
        self._segment_started_at: Optional[datetime] = None
        self._mux_threads: List[threading.Thread] = []
        self._saved: List[Path] = []
        self._error: Optional[BaseException] = None
        self._started_at: Optional[float] = None
        self._rotate_thread: Optional[threading.Thread] = None
        self._temp_dir: Optional[Path] = None
        self._own_temp = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def saved_paths(self) -> List[Path]:
        with self._io_lock:
            return list(self._saved)

    def start(self) -> str:
        if self._recording:
            raise RuntimeError("已在螢幕錄影中。")
        if self._recorder.is_recording:
            raise RuntimeError("請先停止目前的錄音。")

        if self._backend is None:
            self._backend = FFmpegBackend()

        self._stop_event.clear()
        self._error = None
        self._saved = []
        self._segment_index = 0
        self._mux_threads = []
        self._grab = None
        self._wav = None
        self._video_path = None

        if self._temp_dir_arg is not None:
            self._temp_dir = self._temp_dir_arg
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            self._own_temp = False
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="musicrecord_screen_"))
            self._own_temp = True

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._recorder.keep_chunks = False
        self._recorder.chunk_sink = self._on_audio_chunk

        try:
            device_name = self._recorder.start()
            self._recording = True
            self._started_at = time.monotonic()
            self._begin_segment()
        except Exception:
            self._recording = False
            self._cleanup_failed_start()
            raise

        self._rotate_thread = threading.Thread(
            target=self._rotate_loop,
            name="screen-rotate",
            daemon=True,
        )
        self._rotate_thread.start()
        return device_name

    def stop(self) -> List[Path]:
        if not self._recording and self._grab is None:
            return self.saved_paths

        self._recording = False
        self._stop_event.set()
        if self._rotate_thread is not None:
            self._rotate_thread.join(timeout=8.0)
            self._rotate_thread = None

        with self._rotate_lock:
            video, audio, dest = self._finalize_current()

        try:
            self._recorder.stop()
        except Exception:
            pass
        self._recorder.chunk_sink = None
        self._recorder.keep_chunks = True

        if video is not None:
            self._mux_and_notify(video, audio, dest)

        for thread in list(self._mux_threads):
            thread.join(timeout=180)

        saved = self.saved_paths
        self._cleanup_temp()
        self._started_at = None
        if self._error is not None and not saved:
            err = self._error
            self._error = None
            raise RuntimeError(f"螢幕錄影失敗：{err}") from err
        return saved

    def _on_audio_chunk(self, arr) -> None:
        with self._io_lock:
            if self._wav is not None:
                self._wav.write(arr)

    def _begin_segment(self) -> None:
        assert self._temp_dir is not None
        assert self._backend is not None
        self._segment_index += 1
        video_path = self._temp_dir / f"video_{self._segment_index:03d}.mp4"
        wav_path = self._temp_dir / f"audio_{self._segment_index:03d}.wav"
        with self._io_lock:
            self._wav = StreamingWavWriter(
                wav_path,
                samplerate=self._recorder.samplerate,
                channels=self._recorder.channels,
            )
            self._video_path = video_path
            self._segment_started_at = self._clock()
        try:
            self._grab = self._backend.start_grab(video_path)
        except Exception:
            with self._io_lock:
                if self._wav is not None:
                    self._wav.close()
                    self._wav = None
                self._video_path = None
            raise

    def _finalize_current(
        self,
    ) -> tuple[Optional[Path], Optional[Path], Path]:
        grab = self._grab
        self._grab = None
        if grab is not None:
            try:
                grab.request_stop()
                grab.wait(timeout=25)
            except Exception:
                pass

        with self._io_lock:
            wav_path = self._wav.close() if self._wav is not None else None
            self._wav = None
            video_path = self._video_path
            self._video_path = None
            started = self._segment_started_at or self._clock()

        dest = unique_output_path(self._output_dir, default_screen_filename(started))
        return video_path, wav_path, dest

    def _spawn_mux(
        self,
        video: Optional[Path],
        audio: Optional[Path],
        dest: Path,
    ) -> None:
        thread = threading.Thread(
            target=self._mux_and_notify,
            args=(video, audio, dest),
            name="screen-mux",
            daemon=True,
        )
        self._mux_threads.append(thread)
        thread.start()

    def _mux_and_notify(
        self,
        video: Optional[Path],
        audio: Optional[Path],
        dest: Path,
    ) -> None:
        err: Optional[BaseException] = None
        saved: Optional[Path] = None
        try:
            if video is None or not video.exists() or video.stat().st_size == 0:
                raise RuntimeError("沒有錄到畫面。")
            assert self._backend is not None
            self._backend.mux(video, audio, dest)
            saved = dest
            with self._io_lock:
                self._saved.append(dest)
        except BaseException as exc:  # noqa: BLE001
            err = exc
            with self._io_lock:
                self._error = exc
        finally:
            for path in (video, audio):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            callback = self.on_segment_done
            if callback is not None:
                try:
                    callback(saved, err)
                except Exception:
                    pass

    def _rotate_loop(self) -> None:
        while not self._stop_event.is_set() and self._recording:
            if self._started_at is None:
                break
            deadline = self._started_at + self._segment_seconds * self._segment_index
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._stop_event.wait(timeout=min(remaining, 0.25))
                continue
            if self._stop_event.is_set() or not self._recording:
                break
            try:
                self._rotate_segment()
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                self._recording = False
                callback = self.on_segment_done
                if callback is not None:
                    try:
                        callback(None, exc)
                    except Exception:
                        pass
                break

    def _rotate_segment(self) -> None:
        with self._rotate_lock:
            if not self._recording or self._stop_event.is_set():
                return
            video, audio, dest = self._finalize_current()
            if self._recording and not self._stop_event.is_set():
                self._begin_segment()
        self._spawn_mux(video, audio, dest)

    def _cleanup_failed_start(self) -> None:
        self._recorder.chunk_sink = None
        self._recorder.keep_chunks = True
        if self._recorder.is_recording:
            try:
                self._recorder.stop()
            except Exception:
                pass
        try:
            self._finalize_current()
        except Exception:
            pass
        self._cleanup_temp()
        self._started_at = None

    def _cleanup_temp(self) -> None:
        if self._own_temp and self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        if self._own_temp:
            self._temp_dir = None
