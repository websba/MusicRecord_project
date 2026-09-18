"""customtkinter UI for system audio recording and screen capture."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, List, Optional

import customtkinter as ctk
import numpy as np

from music_recorder.capture import LoopbackRecorder
from music_recorder.devices import get_default_playback_device
from music_recorder.encoder import save_mp3
from music_recorder.screen import ScreenRecorder


def default_output_dir() -> Path:
    return Path.home() / "Desktop"


def default_downloads_dir() -> Path:
    """Alias kept for older call sites; default output is the Desktop."""
    return default_output_dir()


def format_elapsed(seconds: float) -> str:
    total = int(max(0, seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def default_filename(now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"recording_{stamp}.mp3"


class RecorderApp(ctk.CTk):
    """Main window: audio record / screen record / timer / save."""

    def __init__(
        self,
        recorder: Optional[LoopbackRecorder] = None,
        ask_save_path: Optional[Callable[[str, Path], Optional[str]]] = None,
        output_dir: Optional[Path] = None,
        downloads_dir: Optional[Path] = None,
        screen_recorder: Optional[ScreenRecorder] = None,
    ) -> None:
        super().__init__()
        self.title("系統音訊錄音／錄影")
        self.geometry("420x400")
        self.minsize(380, 360)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self._recorder = recorder or LoopbackRecorder()
        self._ask_save_path = ask_save_path or self._dialog_save_path
        self._output_dir = output_dir or downloads_dir or default_output_dir()
        self._screen = screen_recorder
        self._tick_job: Optional[str] = None
        self._busy = False
        self._stop_result: Optional[tuple[Optional[np.ndarray], Optional[BaseException]]] = None
        self._stop_result_lock = threading.Lock()
        self._screen_stop_result: Optional[tuple[Optional[List[Path]], Optional[BaseException]]] = None

        self._build_ui()
        self._refresh_device_label()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self,
            text="系統音訊錄音／錄影",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, padx=24, pady=(24, 8), sticky="w")

        hint = ctk.CTkLabel(
            self,
            text="錄音擷取播放裝置；螢幕錄影含多螢幕與系統音訊，每小時存到桌面",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray65"),
            wraplength=360,
            justify="left",
        )
        hint.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="w")

        self.device_label = ctk.CTkLabel(
            self,
            text="目前裝置：—",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self.device_label.grid(row=2, column=0, padx=24, pady=(0, 8), sticky="ew")

        self.timer_label = ctk.CTkLabel(
            self,
            text="00:00",
            font=ctk.CTkFont(size=48, weight="bold"),
        )
        self.timer_label.grid(row=3, column=0, padx=24, pady=12)

        self.status_label = ctk.CTkLabel(
            self,
            text="就緒",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray65"),
        )
        self.status_label.grid(row=4, column=0, padx=24, pady=(0, 16), sticky="w")

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=5, column=0, padx=24, pady=(0, 8), sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.record_btn = ctk.CTkButton(
            btn_row,
            text="錄音",
            height=40,
            command=self.start_recording,
            fg_color="#c0392b",
            hover_color="#a93226",
        )
        self.record_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_row,
            text="停止",
            height=40,
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        screen_row = ctk.CTkFrame(self, fg_color="transparent")
        screen_row.grid(row=6, column=0, padx=24, pady=(0, 24), sticky="ew")
        screen_row.grid_columnconfigure((0, 1), weight=1)

        self.screen_record_btn = ctk.CTkButton(
            screen_row,
            text="螢幕錄影",
            height=40,
            command=self.start_screen_recording,
        )
        self.screen_record_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.screen_stop_btn = ctk.CTkButton(
            screen_row,
            text="停止",
            height=40,
            command=self.stop_screen_recording,
            state="disabled",
        )
        self.screen_stop_btn.grid(row=0, column=1, padx=(8, 0), sticky="ew")

    def _dialog_save_path(self, initial_name: str, initial_dir: Path) -> Optional[str]:
        initial_dir.mkdir(parents=True, exist_ok=True)
        return filedialog.asksaveasfilename(
            parent=self,
            title="儲存錄音",
            defaultextension=".mp3",
            filetypes=[("MP3 音訊", "*.mp3"), ("所有檔案", "*.*")],
            initialdir=str(initial_dir),
            initialfile=initial_name,
        )

    def _ensure_screen(self) -> ScreenRecorder:
        if self._screen is None:
            self._screen = ScreenRecorder(
                recorder=self._recorder,
                output_dir=self._output_dir,
                on_segment_done=self._on_segment_done,
            )
        elif getattr(self._screen, "on_segment_done", None) is None:
            self._screen.on_segment_done = self._on_segment_done
        return self._screen

    def _screen_is_recording(self) -> bool:
        return self._screen is not None and bool(self._screen.is_recording)

    def _on_segment_done(
        self,
        path: Optional[Path],
        error: Optional[BaseException],
    ) -> None:
        def apply() -> None:
            if error is not None:
                self.status_label.configure(text="該段存檔失敗")
                messagebox.showerror("螢幕錄影", str(error), parent=self)
                return
            if path is None:
                return
            if self._screen_is_recording():
                self.status_label.configure(text=f"錄影中… 已儲存：{path.name}")
            else:
                self.status_label.configure(text=f"已儲存：{path.name}")

        try:
            self.after(0, apply)
        except Exception:
            pass

    def _refresh_device_label(self) -> None:
        # Avoid calling WASAPI from the UI thread while capturing;
        # COM/WASAPI multi-thread use can crash the process on Windows.
        if self._recorder.is_recording:
            name = self._recorder.device_name or "錄音中…"
        else:
            try:
                name = get_default_playback_device().name
            except Exception:
                name = "無法偵測"
        self.device_label.configure(text=f"目前裝置：{name}")

    def _set_idle_buttons(self) -> None:
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.screen_record_btn.configure(state="normal")
        self.screen_stop_btn.configure(state="disabled")

    def start_recording(self) -> None:
        if self._busy or self._recorder.is_recording or self._screen_is_recording():
            return
        try:
            device_name = self._recorder.start()
        except Exception as exc:
            messagebox.showerror("無法開始錄音", str(exc), parent=self)
            return

        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.screen_record_btn.configure(state="disabled")
        self.screen_stop_btn.configure(state="disabled")
        self.status_label.configure(text="錄音中…")
        self.device_label.configure(text=f"目前裝置：{device_name}")
        self.timer_label.configure(text="00:00")
        self._schedule_tick()

    def stop_recording(self) -> None:
        if self._busy or self._screen_is_recording() or not self._recorder.is_recording:
            return
        self._busy = True
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="處理中…")
        self._cancel_tick()

        with self._stop_result_lock:
            self._stop_result = None

        def worker() -> None:
            pcm: Optional[np.ndarray] = None
            error: Optional[BaseException] = None
            try:
                pcm = self._recorder.stop()
            except BaseException as exc:  # noqa: BLE001
                error = exc
            with self._stop_result_lock:
                self._stop_result = (pcm, error)

        threading.Thread(target=worker, name="stop-encode", daemon=True).start()
        self._poll_stop_result()

    def start_screen_recording(self) -> None:
        if self._busy or self._recorder.is_recording or self._screen_is_recording():
            return
        screen = self._ensure_screen()
        try:
            device_name = screen.start()
        except Exception as exc:
            messagebox.showerror("無法開始螢幕錄影", str(exc), parent=self)
            return

        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.screen_record_btn.configure(state="disabled")
        self.screen_stop_btn.configure(state="normal")
        self.status_label.configure(text="螢幕錄影中…")
        self.device_label.configure(text=f"目前裝置：{device_name}")
        self.timer_label.configure(text="00:00")
        self._schedule_tick()

    def stop_screen_recording(self) -> None:
        if self._busy or not self._screen_is_recording():
            return
        self._busy = True
        self.screen_stop_btn.configure(state="disabled")
        self.status_label.configure(text="處理中…")
        self._cancel_tick()

        with self._stop_result_lock:
            self._screen_stop_result = None

        def worker() -> None:
            paths: Optional[List[Path]] = None
            error: Optional[BaseException] = None
            try:
                assert self._screen is not None
                paths = self._screen.stop()
            except BaseException as exc:  # noqa: BLE001
                error = exc
            with self._stop_result_lock:
                self._screen_stop_result = (paths, error)

        threading.Thread(target=worker, name="stop-screen", daemon=True).start()
        self._poll_screen_stop_result()

    def _poll_stop_result(self) -> None:
        with self._stop_result_lock:
            result = self._stop_result
            if result is not None:
                self._stop_result = None
            else:
                result = None

        if result is not None:
            pcm, error = result
            self._after_stop(pcm, error)
            return

        if self._busy:
            self.after(50, self._poll_stop_result)

    def _poll_screen_stop_result(self) -> None:
        with self._stop_result_lock:
            result = self._screen_stop_result
            if result is not None:
                self._screen_stop_result = None
            else:
                result = None

        if result is not None:
            paths, error = result
            self._after_screen_stop(paths, error)
            return

        if self._busy:
            self.after(50, self._poll_screen_stop_result)

    def _after_stop(
        self,
        pcm: Optional[np.ndarray],
        error: Optional[BaseException],
    ) -> None:
        self._busy = False
        self._set_idle_buttons()
        self._refresh_device_label()

        if error is not None:
            self.status_label.configure(text="錄音失敗")
            messagebox.showerror("錄音失敗", str(error), parent=self)
            return

        if pcm is None or pcm.size == 0:
            self.status_label.configure(text="沒有錄到音訊")
            messagebox.showwarning(
                "沒有音訊",
                "沒有錄到任何聲音。請確認電腦正在播放音樂後再試。",
                parent=self,
            )
            return

        path = self._ask_save_path(default_filename(), self._output_dir)
        if not path:
            self.status_label.configure(text="已取消存檔")
            return

        try:
            save_mp3(path, pcm, self._recorder.samplerate)
        except Exception as exc:
            self.status_label.configure(text="存檔失敗")
            messagebox.showerror("存檔失敗", str(exc), parent=self)
            return

        self.status_label.configure(text=f"已儲存：{Path(path).name}")
        self.timer_label.configure(text="00:00")

    def _after_screen_stop(
        self,
        paths: Optional[List[Path]],
        error: Optional[BaseException],
    ) -> None:
        self._busy = False
        self._set_idle_buttons()
        self._refresh_device_label()
        self.timer_label.configure(text="00:00")

        if error is not None:
            self.status_label.configure(text="螢幕錄影失敗")
            messagebox.showerror("螢幕錄影失敗", str(error), parent=self)
            return

        saved = [p for p in (paths or []) if p is not None]
        if not saved:
            self.status_label.configure(text="沒有錄到畫面")
            messagebox.showwarning(
                "沒有畫面",
                "沒有產生錄影檔。請再試一次。",
                parent=self,
            )
            return

        if len(saved) == 1:
            self.status_label.configure(text=f"已儲存：{saved[0].name}")
        else:
            self.status_label.configure(text=f"已儲存 {len(saved)} 個檔案到桌面")

    def _schedule_tick(self) -> None:
        self._cancel_tick()
        self._tick()

    def _tick(self) -> None:
        screen_on = self._screen_is_recording()
        audio_on = self._recorder.is_recording
        if not screen_on and not audio_on:
            return
        if screen_on and self._screen is not None:
            seconds = self._screen.elapsed_seconds
        else:
            seconds = self._recorder.elapsed_seconds
        self.timer_label.configure(text=format_elapsed(seconds))
        name = self._recorder.device_name
        if name:
            self.device_label.configure(text=f"目前裝置：{name}")
        self._tick_job = self.after(200, self._tick)

    def _cancel_tick(self) -> None:
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None

    def _on_close(self) -> None:
        self._cancel_tick()
        try:
            if self._screen_is_recording() and self._screen is not None:
                self._screen.stop()
            elif self._recorder.is_recording or self._busy:
                self._recorder.stop()
        except Exception:
            pass
        self.destroy()


def run_app() -> None:
    app = RecorderApp()
    app.mainloop()
