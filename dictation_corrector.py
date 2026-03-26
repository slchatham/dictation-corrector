#!/usr/bin/env python3
"""
Dictation Corrector v2
Pipeline: Mic → Parakeet ASR (nvidia/parakeet-tdt-0.6b-v3) → Qwen3.5:4b → floating window

Audio modes:
  STREAMING  continuous capture in 4s chunks, Cmd+Shift+K triggers correction
  FILE       Cmd+Shift+R start/stop, batch transcription, automatic correction

Shortcuts:
  Cmd+Shift+K  LLM correction on current buffer
  Cmd+Shift+R  start/stop recording (FILE mode)

Dependencies:
  pip install nemo_toolkit[asr] sounddevice soundfile numpy httpx pynput rumps

Usage:
  python dictation_corrector.py
"""

from __future__ import annotations
import datetime
import json
import os
import queue
import sys
import tempfile
import threading

import numpy as np
import httpx
import pyperclip


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen3.5:4b"
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE    = 16000
CHUNK_SECONDS        = 4
IMPORT_TARGET_SECONDS  = 90   # target chunk duration for import
IMPORT_SILENCE_SEARCH  = 15   # look for silence in the last N seconds of each target chunk

SYSTEM_PROMPT = """\
Tu es un correcteur de dictée vocale pour un auteur bilingue français/anglais.
Le texte t'est envoyé tel que capturé par un ASR. L'auteur alterne librement entre les deux langues.

Corrige UNIQUEMENT ces trois cas, et seulement si tu en es certain à plus de 95% :
1. Mot manifestement phonétisé par l'ASR (ex : "baguette" → "backlog", "dîtes" → "data")
   — si le mot est ambigu ou pourrait être intentionnel, laisse-le tel quel
2. Erreur de transcription phonétique évidente dans la langue en cours
   ex : "pansant" → "pensant",  "their" → "there"
3. Mot de ponctuation dicté oralement
   ex : "virgule" → ",",  "point" → ".",  "à la ligne" → "\\n",  "tiret" → "–"

NE JAMAIS :
- traduire une phrase d'une langue à l'autre (l'anglais intentionnel reste anglais, le français reste français)
- ajouter un mot qui n'est pas dans le texte source
- reformuler, compléter ou corriger le fond
- modifier la ponctuation existante
- corriger un mot si tu n'es pas sûr à plus de 95%

Retourne uniquement le texte corrigé, sans commentaire ni explication.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    ts     = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    prefix = {"INFO": "·", "OK": "✓", "WARN": "⚠", "ERR": "✗"}.get(level, "·")
    print(f"[{ts}] {prefix} {msg}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# <think>…</think> filter  (qwen3 reasoning tokens)
# ─────────────────────────────────────────────────────────────────────────────

class _ThinkFilter:
    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in  = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._in:
                p = self._buf.find(self._CLOSE)
                if p >= 0:
                    self._buf = self._buf[p + len(self._CLOSE):]
                    self._in  = False
                else:
                    self._buf = ""
                    break
            else:
                p = self._buf.find(self._OPEN)
                if p >= 0:
                    out.append(self._buf[:p])
                    self._buf = self._buf[p + len(self._OPEN):]
                    self._in  = True
                else:
                    for i in range(1, len(self._OPEN)):
                        if self._buf.endswith(self._OPEN[:i]):
                            out.append(self._buf[:-i])
                            self._buf = self._buf[-i:]
                            break
                    else:
                        out.append(self._buf)
                        self._buf = ""
                    break
        return "".join(out)

    def flush(self) -> str:
        val, self._buf = ("" if self._in else self._buf), ""
        return val


# ─────────────────────────────────────────────────────────────────────────────
# UI MODE  ──  python dictation_corrector.py --ui
# Persistent tkinter window.
# Receives updates from daemon via stdin (JSON-lines).
# Sends commands to daemon via stdout (JSON-lines).
# ─────────────────────────────────────────────────────────────────────────────

def _run_ui_mode() -> None:
    import tkinter as tk

    log("UI mode started")

    # ── Catppuccin Mocha palette (color-blind friendly) ───────────────────────
    # Red and green replaced by orange and blue — distinguishable under
    # deuteranopia/protanopia. Icons provide shape cues independent of color.
    C = dict(
        bg      = "#1e1e2e",
        surface = "#313244",
        overlay = "#45475a",
        text    = "#cdd6f4",
        sub     = "#a6adc8",
        dim     = "#585b70",
        blue    = "#89b4fa",   # active / info
        orange  = "#fab387",   # stop / danger / quit  (replaces red)
        yellow  = "#f9e2af",
        teal    = "#94e2d5",
    )

    # ── Send a command to the daemon via stdout ───────────────────────────────
    def send_cmd(cmd: dict) -> None:
        try:
            sys.stdout.write(json.dumps(cmd) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            log(f"send_cmd: {exc}", "ERR")

    # ── Window root ──────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Dictation Corrector v2")
    root.configure(bg=C["bg"])
    root.attributes("-topmost", True)
    root.geometry("700x620")
    root.resizable(True, True)

    # ── Dynamic variables ─────────────────────────────────────────────────────
    mode_var   = tk.StringVar(value="STREAMING")
    status_var = tk.StringVar(value="⟳  Loading Parakeet…")
    rec_var    = tk.StringVar(value="⏺  Record")

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg=C["bg"])
    hdr.pack(fill="x", padx=14, pady=(14, 6))

    tk.Label(hdr, text="Dictation Corrector v2",
             bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 10)).pack(side="left")

    def toggle_mode() -> None:
        new = "FICHIER" if mode_var.get() == "STREAMING" else "STREAMING"
        send_cmd({"cmd": "switch_mode", "mode": new})

    mode_btn = tk.Button(
        hdr, textvariable=mode_var,
        bg=C["surface"], fg=C["teal"], activebackground=C["overlay"],
        font=("Helvetica Neue", 10, "bold"), bd=0, relief="flat",
        padx=10, pady=4, command=toggle_mode,
    )
    mode_btn.pack(side="right")
    tk.Label(hdr, text="Mode:", bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 10)).pack(side="right", padx=(0, 4))

    # ── Copy helper ───────────────────────────────────────────────────────────
    def _copy_widget(w: tk.Text, btn: tk.Button) -> None:
        w.config(state="normal")
        text = w.get("1.0", "end").strip()
        w.config(state="disabled")
        if text:
            pyperclip.copy(text)
            btn.config(text="✓")
            root.after(2000, lambda: btn.config(text="Copy"))

    # ── Raw Parakeet transcription ────────────────────────────────────────────
    hdr_raw = tk.Frame(root, bg=C["bg"])
    hdr_raw.pack(fill="x", padx=14, pady=(4, 2))
    tk.Label(hdr_raw, text="PARAKEET TRANSCRIPTION",
             bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 9)).pack(side="left")

    f_raw = tk.Frame(root, bg=C["surface"])
    f_raw.pack(fill="x", padx=12, pady=(0, 8))

    w_raw = tk.Text(
        f_raw, wrap="word", bg=C["surface"], fg=C["dim"],
        font=("Helvetica Neue", 12), bd=0, relief="flat",
        height=5, padx=10, pady=8, state="disabled",
    )
    w_raw.pack(fill="both")

    raw_copy_btn = tk.Button(
        hdr_raw, text="Copy",
        bg=C["surface"], fg=C["sub"], activebackground=C["overlay"],
        font=("Helvetica Neue", 9), bd=0, relief="flat", padx=8, pady=2,
        command=lambda: _copy_widget(w_raw, raw_copy_btn),
    )
    raw_copy_btn.pack(side="right")

    # ── Qwen3.5 correction ───────────────────────────────────────────────────
    hdr_corr = tk.Frame(root, bg=C["bg"])
    hdr_corr.pack(fill="x", padx=14, pady=(0, 2))
    tk.Label(hdr_corr, text="QWEN3.5 CORRECTION",
             bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 9)).pack(side="left")

    f_corr = tk.Frame(root, bg=C["surface"])
    f_corr.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    w_corr = tk.Text(
        f_corr, wrap="word", bg=C["surface"], fg=C["text"],
        font=("Helvetica Neue", 13), bd=0, relief="flat",
        height=9, padx=10, pady=8, state="disabled",
    )
    w_corr.pack(fill="both", expand=True)

    corr_copy_btn = tk.Button(
        hdr_corr, text="Copy",
        bg=C["surface"], fg=C["sub"], activebackground=C["overlay"],
        font=("Helvetica Neue", 9), bd=0, relief="flat", padx=8, pady=2,
        command=lambda: _copy_widget(w_corr, corr_copy_btn),
    )
    corr_copy_btn.pack(side="right")

    # ── Status bar ────────────────────────────────────────────────────────────
    tk.Label(root, textvariable=status_var,
             bg=C["bg"], fg=C["blue"],
             font=("Helvetica Neue", 11)).pack(pady=(2, 4))

    # ── Buttons ───────────────────────────────────────────────────────────────
    bf = tk.Frame(root, bg=C["bg"])
    bf.pack(fill="x", padx=12, pady=(0, 14))

    # Record button — visible only in FILE mode
    rec_btn = tk.Button(
        bf, textvariable=rec_var,
        bg=C["surface"], fg=C["text"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "record_toggle"}),
    )

    tk.Button(
        bf, text="Correct now",
        bg=C["teal"], fg=C["bg"], activebackground=C["sub"],
        font=("Helvetica Neue", 11, "bold"), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "correct_now"}),
    ).pack(side="left", padx=(0, 8))

    mute_var = tk.StringVar(value="🎙 Active")
    mute_btn = tk.Button(
        bf, textvariable=mute_var,
        bg=C["surface"], fg=C["blue"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "mute_toggle"}),
    )
    mute_btn.pack(side="left", padx=(0, 8))

    def _pick_file() -> None:
        from tkinter import filedialog
        root.lift()
        root.focus_force()
        path = filedialog.askopenfilename(
            parent=root,
            title="Import audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            send_cmd({"cmd": "import_file", "path": path})

    tk.Button(
        bf, text="Import…",
        bg=C["surface"], fg=C["sub"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=_pick_file,
    ).pack(side="left", padx=(0, 8))

    tk.Button(
        bf, text="Clear",
        bg=C["overlay"], fg=C["sub"], activebackground=C["dim"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "clear"}),
    ).pack(side="right", padx=(0, 8))

    def _exit() -> None:
        send_cmd({"cmd": "exit"})
        root.after(200, root.destroy)

    tk.Button(
        bf, text="Quit",
        bg=C["orange"], fg=C["bg"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11, "bold"), bd=0, relief="flat", padx=12, pady=7,
        command=_exit,
    ).pack(side="right", padx=(0, 8))

    # ── Widget helpers (main thread) ──────────────────────────────────────────
    def _set_widget(w: tk.Text, text: str) -> None:
        w.config(state="normal")
        w.delete("1.0", "end")
        w.insert("1.0", text)
        w.config(state="disabled")
        w.see("end")

    def _append_widget(w: tk.Text, text: str) -> None:
        w.config(state="normal")
        w.insert("end", text)
        w.config(state="disabled")
        w.see("end")

    # ── Incoming message handler ──────────────────────────────────────────────
    def handle_msg(msg: dict) -> None:
        t = msg.get("t")
        v = msg.get("v")
        if t == "status":
            status_var.set(v)
        elif t == "mode":
            mode_var.set(v)
            if v == "FICHIER":
                rec_btn.pack(side="left", padx=(0, 8))
            else:
                rec_btn.pack_forget()
        elif t == "transcript":
            _set_widget(w_raw, v)
        elif t == "corrected_chunk":
            _append_widget(w_corr, v)
        elif t == "clear_corrected":
            _set_widget(w_corr, "")
        elif t == "recording":
            if v:
                rec_var.set("⏹  Stop")
                rec_btn.config(bg=C["orange"], fg=C["bg"])
            else:
                rec_var.set("⏺  Record")
                rec_btn.config(bg=C["surface"], fg=C["text"])
        elif t == "muted":
            if v:
                mute_var.set("🔇 Muted")
                mute_btn.config(bg=C["overlay"], fg=C["dim"])
            else:
                mute_var.set("🎙 Active")
                mute_btn.config(bg=C["surface"], fg=C["blue"])

    # ── Stdin reader (background thread) ─────────────────────────────────────
    def read_stdin() -> None:
        import io
        reader = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
        try:
            for raw in reader:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                    root.after(0, lambda m=msg: handle_msg(m))
                except Exception as exc:
                    log(f"stdin parse: {exc}", "ERR")
            # EOF: daemon stopped externally
            log("Stdin closed (daemon stopped), closing UI")
            root.after(0, root.destroy)
        except (ValueError, OSError):
            pass  # stdin closed after Quit — normal exit path

    threading.Thread(target=read_stdin, daemon=True).start()

    # ── Center window ─────────────────────────────────────────────────────────
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h   = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")

    log("Tkinter window opened")
    root.mainloop()
    log("Tkinter window closed")
    try:
        sys.stdin.buffer.close()   # unblock read_stdin thread before interpreter shutdown
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# DAEMON MODE  ──  python dictation_corrector.py  (default)
# ─────────────────────────────────────────────────────────────────────────────

def _run_daemon_mode() -> None:
    import subprocess
    import rumps
    from pynput import keyboard
    import sounddevice as sd
    import atexit

    # ── Shared state (mutable lists for closures) ─────────────────────────────
    _mode      = ["STREAMING"]   # "STREAMING" | "FILE"
    _recording = [False]
    _transcript = [""]           # accumulated transcription buffer

    # ── UIBridge — bidirectional IPC with the tkinter subprocess ─────────────
    _ui_cmd_handler = [None]     # set later to break circular dependency

    class _UIBridge:
        def __init__(self) -> None:
            self._proc: subprocess.Popen | None = None
            self._lock   = threading.Lock()
            self._reader: threading.Thread | None = None
            atexit.register(self._cleanup)

        def _cleanup(self) -> None:
            proc = None
            with self._lock:
                if self._proc and self._proc.poll() is None:
                    proc = self._proc
            if proc is None:
                return
            log("Closing UI subprocess…")
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            try:
                proc.stdout.close()   # unblocks _read_commands thread
            except Exception:
                pass
            if self._reader and self._reader.is_alive():
                self._reader.join(timeout=1)

        def ensure_alive(self) -> None:
            with self._lock:
                if self._proc is None or self._proc.poll() is not None:
                    log("Starting UI subprocess…")
                    self._proc = subprocess.Popen(
                        [sys.executable, __file__, "--ui"],
                        stdin  = subprocess.PIPE,
                        stdout = subprocess.PIPE,
                        stderr = sys.stderr,
                    )
                    log(f"UI subprocess PID {self._proc.pid}", "OK")
                    self._reader = threading.Thread(
                        target = self._read_commands,
                        args   = (self._proc,),
                        daemon = True,
                    )
                    self._reader.start()

        def send(self, msg: dict) -> None:
            with self._lock:
                if self._proc is None or self._proc.poll() is not None:
                    return
                try:
                    self._proc.stdin.write((json.dumps(msg) + "\n").encode())
                    self._proc.stdin.flush()
                except Exception as exc:
                    log(f"UIBridge.send: {exc}", "ERR")

        def _read_commands(self, proc: subprocess.Popen) -> None:
            import io
            reader = io.TextIOWrapper(proc.stdout, encoding="utf-8")
            try:
                for raw in reader:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        cmd = json.loads(raw)
                        log(f"UI command: {cmd}")
                        handler = _ui_cmd_handler[0]
                        if handler:
                            handler(cmd)
                    except Exception as exc:
                        log(f"UI cmd parse: {exc}", "ERR")
            except (ValueError, OSError):
                pass  # pipe closed during shutdown

    ui = _UIBridge()

    # ── Parakeet engine ───────────────────────────────────────────────────────
    class _ParakeetEngine:
        def __init__(self) -> None:
            self._model  = None
            self._device = "cpu"
            self._lock   = threading.Lock()
            self.ready   = threading.Event()

        def load_async(self, on_ready=None) -> None:
            threading.Thread(target=self._load, args=(on_ready,), daemon=True).start()

        @staticmethod
        def _silence_nemo_loggers() -> None:
            """
            NeMo resets its logging on every operation (from_pretrained, transcribe…).
            Walk ALL registered loggers and force ERROR level on NeMo/Lightning ones.
            Call after model load AND before each transcription.
            """
            import logging
            _PREFIXES = ("nemo", "lightning", "pytorch_lightning",
                         "torch.distributed", "nv_one_logger", "one_logger")
            for name, logger in list(logging.Logger.manager.loggerDict.items()):
                if isinstance(logger, logging.Logger) and any(
                    name.startswith(p) for p in _PREFIXES
                ):
                    logger.setLevel(logging.ERROR)
                    logger.propagate = False

        def _load(self, on_ready=None) -> None:
            import warnings
            warnings.filterwarnings("ignore")   # suppress PyTorch/NeMo warnings at init
            self._silence_nemo_loggers()        # first pass before NeMo import
            try:
                import torch
                import nemo.collections.asr as nemo_asr

                log(f"Loading {PARAKEET_MODEL}…")
                ui.send({"t": "status", "v": "⟳  Loading Parakeet…"})

                self._model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL)

                if torch.backends.mps.is_available():
                    try:
                        self._model = self._model.to("mps")
                        self._device = "mps"
                        log("Parakeet on MPS (Apple Silicon)", "OK")
                    except Exception as exc:
                        log(f"MPS unavailable ({exc}), falling back to CPU", "WARN")
                        self._device = "cpu"
                else:
                    log("MPS not available, Parakeet on CPU")

                self._model.eval()
                self._silence_nemo_loggers()    # second pass after from_pretrained
                log(f"Parakeet ready on {self._device}", "OK")
                ui.send({"t": "status", "v": "○  Idle"})
                self.ready.set()
                if on_ready:
                    on_ready()

            except Exception as exc:
                log(f"Parakeet load error: {exc}", "ERR")
                ui.send({"t": "status", "v": f"✗  Parakeet : {str(exc)[:80]}"})

        def transcribe(self, audio_np: np.ndarray) -> str:
            if self._model is None or len(audio_np) == 0:
                return ""
            with self._lock:
                import soundfile as sf
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                fname = tmp.name
                try:
                    sf.write(fname, audio_np, SAMPLE_RATE)
                    tmp.close()
                    self._silence_nemo_loggers()    # NeMo recreates loggers on every call
                    result = self._model.transcribe([fname], verbose=False)
                    # NeMo may return list[str] or tuple(list[str], ...)
                    if isinstance(result, tuple):
                        result = result[0]
                    text = result[0] if result else ""
                    # Some versions return a Hypothesis object
                    if hasattr(text, "text"):
                        text = text.text
                    return str(text).strip()
                except Exception as exc:
                    log(f"transcribe() error: {exc}", "ERR")
                    return ""
                finally:
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass

    parakeet = _ParakeetEngine()

    # ── Audio engine (sounddevice) ────────────────────────────────────────────
    _transcription_queue: queue.Queue = queue.Queue()

    class _AudioEngine:
        def __init__(self) -> None:
            self._stream          = None
            self._chunk_buf:  list[np.ndarray] = []
            self._chunk_n     = 0
            self._rec_buf:    list[np.ndarray] = []
            self._streaming   = False
            self._muted       = False
            self._lock        = threading.Lock()

        def start_stream(self) -> None:
            try:
                self._stream = sd.InputStream(
                    samplerate = SAMPLE_RATE,
                    channels   = 1,
                    dtype      = "float32",
                    blocksize  = int(SAMPLE_RATE * 0.1),   # 100 ms blocks
                    callback   = self._callback,
                )
                self._stream.start()
                log("Audio stream opened", "OK")
            except Exception as exc:
                log(f"sounddevice : {exc}", "ERR")
                if "permission" in str(exc).lower() or "access" in str(exc).lower():
                    ui.send({"t": "status",
                             "v": "✗  Microphone permission denied — allow in System Settings"})
                else:
                    ui.send({"t": "status", "v": f"✗  Micro : {str(exc)[:80]}"})

        def enable_streaming(self, active: bool) -> None:
            with self._lock:
                self._streaming = active
                if not active:
                    self._chunk_buf.clear()
                    self._chunk_n = 0

        def start_recording(self) -> None:
            with self._lock:
                self._rec_buf.clear()
                _recording[0] = True

        def stop_recording(self) -> np.ndarray:
            with self._lock:
                _recording[0] = False
                if not self._rec_buf:
                    return np.array([], dtype=np.float32)
                return np.concatenate(self._rec_buf)

        def set_muted(self, muted: bool) -> None:
            with self._lock:
                self._muted = muted
                if muted:
                    self._chunk_buf.clear()
                    self._chunk_n = 0
            log(f"Microphone {'muted' if muted else 'active'}", "OK")

        def close(self) -> None:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass

        def _callback(self, indata, frames, time, status) -> None:
            if status:
                log(f"Audio status: {status}", "WARN")
            if self._muted:
                return
            mono = indata[:, 0].copy()

            if _mode[0] == "FICHIER" and _recording[0]:
                with self._lock:
                    self._rec_buf.append(mono)

            elif _mode[0] == "STREAMING" and self._streaming:
                with self._lock:
                    self._chunk_buf.append(mono)
                    self._chunk_n += len(mono)
                    if self._chunk_n >= SAMPLE_RATE * CHUNK_SECONDS:
                        chunk = np.concatenate(self._chunk_buf)
                        self._chunk_buf.clear()
                        self._chunk_n = 0
                        _transcription_queue.put(chunk)

    audio = _AudioEngine()
    atexit.register(audio.close)

    # ── Transcription worker (dedicated thread) ───────────────────────────────
    def _transcription_worker() -> None:
        log("Transcription worker started")
        while True:
            chunk = _transcription_queue.get()
            if chunk is None:
                break
            if not parakeet.ready.is_set():
                log("Parakeet not ready yet, chunk dropped", "WARN")
                continue
            dur = len(chunk) / SAMPLE_RATE
            log(f"Transcribing chunk {dur:.1f}s…")
            ui.send({"t": "status", "v": "◎  Transcription…"})
            text = parakeet.transcribe(chunk)
            if text:
                sep = " " if _transcript[0] else ""
                _transcript[0] += sep + text
                log(f"→ '{text[:80]}'")
                ui.send({"t": "transcript", "v": _transcript[0]})
            ui.send({"t": "status", "v": "○  Idle"})

    threading.Thread(target=_transcription_worker, daemon=True).start()

    # ── LLM correction ────────────────────────────────────────────────────────
    _correction_lock = threading.Lock()

    def _trigger_correction(text: str | None = None) -> None:
        text = text or _transcript[0]
        if not text.strip():
            log("Buffer empty, correction skipped", "WARN")
            ui.send({"t": "status", "v": "⚠  Nothing to correct (buffer empty)"})
            return
        threading.Thread(target=_run_correction, args=(text,), daemon=True).start()

    def _run_correction(text: str, clear: bool = True) -> None:
        if not _correction_lock.acquire(blocking=False):
            log("Correction already in progress", "WARN")
            return
        try:
            log(f"LLM correction ({len(text)} chars)…")
            ui.send({"t": "status", "v": "✦  Correcting…"})
            if clear:
                ui.send({"t": "clear_corrected"})
            flt = _ThinkFilter()
            with httpx.Client(timeout=120.0) as cli:
                with cli.stream("POST", OLLAMA_URL, json={
                    "model":  OLLAMA_MODEL,
                    "prompt": text,
                    "system": SYSTEM_PROMPT,
                    "stream": True,
                    "think":  False,
                }) as resp:
                    resp.raise_for_status()
                    log(f"Ollama HTTP {resp.status_code}, streaming…", "OK")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data  = json.loads(line)
                        chunk = flt.feed(data.get("response", ""))
                        if chunk:
                            ui.send({"t": "corrected_chunk", "v": chunk})
                        if data.get("done"):
                            tokens = data.get("eval_count", "?")
                            dur    = data.get("eval_duration", 0) / 1e9
                            log(f"Correction done — {tokens} tokens in {dur:.1f}s", "OK")
                            break
                    tail = flt.flush()
                    if tail:
                        ui.send({"t": "corrected_chunk", "v": tail})
            ui.send({"t": "status", "v": "✓  Correction done"})
        except httpx.ConnectError:
            log(f"Ollama unreachable at {OLLAMA_URL}", "ERR")
            ui.send({"t": "status", "v": "✗  Ollama unreachable — run: ollama serve"})
        except Exception as exc:
            log(f"Correction error: {exc}", "ERR")
            ui.send({"t": "status", "v": f"✗  {str(exc)[:80]}"})
        finally:
            _correction_lock.release()

    # ── Recording toggle (FILE mode) ──────────────────────────────────────────
    def _record_toggle() -> None:
        if _mode[0] != "FICHIER":
            log("Record ignored outside FILE mode", "WARN")
            return
        if not _recording[0]:
            log("Starting recording…")
            _transcript[0] = ""
            ui.send({"t": "transcript", "v": ""})
            ui.send({"t": "clear_corrected"})
            audio.start_recording()
            ui.send({"t": "recording", "v": True})
            ui.send({"t": "status",    "v": "⏺  Recording…"})
        else:
            log("Stopping recording…")
            raw_audio = audio.stop_recording()
            ui.send({"t": "recording", "v": False})
            if len(raw_audio) < SAMPLE_RATE * 0.5:
                log("Recording too short (<0.5s)", "WARN")
                ui.send({"t": "status", "v": "⚠  Recording too short"})
                return
            dur = len(raw_audio) / SAMPLE_RATE
            log(f"Batch transcription {dur:.1f}s…")
            ui.send({"t": "status", "v": f"◎  Batch transcription ({dur:.1f}s)…"})

            def _batch() -> None:
                text = parakeet.transcribe(raw_audio)
                if text:
                    _transcript[0] = text
                    log(f"Batch → '{text[:80]}'", "OK")
                    ui.send({"t": "transcript", "v": text})
                    _trigger_correction(text)
                else:
                    log("Batch transcription empty", "WARN")
                    ui.send({"t": "status", "v": "⚠  Transcription empty"})

            threading.Thread(target=_batch, daemon=True).start()

    # ── Audio file import (WAV / MP3 / M4A / FLAC / OGG) ─────────────────────
    def _import_file(path: str) -> None:
        import librosa
        name = os.path.basename(path)
        log(f"Importing {name}…")
        ui.send({"t": "status",     "v": f"⟳  Loading {name}…"})
        ui.send({"t": "transcript", "v": ""})
        ui.send({"t": "clear_corrected"})
        _transcript[0] = ""
        try:
            audio_np, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            total_dur = len(audio_np) / SAMPLE_RATE
            log(f"Loaded {name}: {total_dur/60:.1f} min")

            # ── Silence-based chunking ─────────────────────────────────────
            def _silence_chunks(audio: np.ndarray) -> list[tuple[int, int]]:
                """
                Split audio into chunks of ~IMPORT_TARGET_SECONDS by finding the
                last silence within the final IMPORT_SILENCE_SEARCH seconds of
                each target boundary. Falls back to hard cut if no silence found.
                """
                target  = SAMPLE_RATE * IMPORT_TARGET_SECONDS
                search  = SAMPLE_RATE * IMPORT_SILENCE_SEARCH
                frame   = int(SAMPLE_RATE * 0.05)   # 50 ms energy frames
                thr_rms = 10 ** (-38 / 20)           # –38 dBFS silence threshold

                bounds: list[tuple[int, int]] = []
                pos = 0
                while pos < len(audio):
                    end = pos + target
                    if end >= len(audio):
                        bounds.append((pos, len(audio)))
                        break
                    # search window: [end - search, end]
                    w_start  = max(pos, end - search)
                    segment  = audio[w_start:end]
                    n_frames = len(segment) // frame
                    cut = end   # default: hard cut
                    if n_frames > 0:
                        rms = np.array([
                            np.sqrt(np.mean(segment[f*frame:(f+1)*frame] ** 2))
                            for f in range(n_frames)
                        ])
                        silent = np.where(rms < thr_rms)[0]
                        if len(silent):
                            cut = w_start + int(silent[-1]) * frame
                    bounds.append((pos, cut))
                    pos = cut
                return bounds

            chunks_bounds = _silence_chunks(audio_np)
            n_chunks      = len(chunks_bounds)

            def _transcribe_timed(chunk: np.ndarray, label: str) -> str:
                """Transcribe while updating status bar with elapsed seconds."""
                import time
                stop = threading.Event()
                def _tick() -> None:
                    t0 = time.time()
                    while not stop.wait(1.0):
                        ui.send({"t": "status",
                                 "v": f"◎  {label} ({int(time.time() - t0)}s)…"})
                threading.Thread(target=_tick, daemon=True).start()
                try:
                    return parakeet.transcribe(chunk)
                finally:
                    stop.set()

            for i, (start, end) in enumerate(chunks_bounds):
                chunk     = audio_np[start:end]
                chunk_dur = len(chunk) / SAMPLE_RATE
                t_start   = start / SAMPLE_RATE / 60
                t_end     = end   / SAMPLE_RATE / 60
                log(f"Chunk {i+1}/{n_chunks}: {t_start:.1f}–{t_end:.1f} min ({chunk_dur:.0f}s)")
                label = f"Chunk {i+1}/{n_chunks} — Parakeet ({chunk_dur:.0f}s)"

                text = _transcribe_timed(chunk, label)
                if not text:
                    log(f"Chunk {i+1}/{n_chunks}: empty transcription", "WARN")
                    continue

                log(f"Chunk {i+1}/{n_chunks} transcribed → '{text[:60]}'", "OK")
                sep = "\n\n" if _transcript[0] else ""
                _transcript[0] += sep + text
                ui.send({"t": "transcript", "v": _transcript[0]})

                # Separator between corrected chunks in the correction zone
                if i > 0:
                    ui.send({"t": "corrected_chunk", "v": "\n\n"})

                ui.send({"t": "status",
                         "v": f"✦  Chunk {i+1}/{n_chunks} — Correcting…"})
                _run_correction(text, clear=(i == 0))

            ui.send({"t": "status",
                     "v": f"✓  Import done — {n_chunks} chunk(s), {total_dur/60:.1f} min"})

        except Exception as exc:
            log(f"Import error: {exc}", "ERR")
            ui.send({"t": "status", "v": f"✗  Import error: {str(exc)[:60]}"})

    # ── Mode switch ───────────────────────────────────────────────────────────
    _mode_item_ref = [None]   # reference to rumps MenuItem, set later

    def _set_mode(new_mode: str) -> None:
        if _mode[0] == new_mode:
            return
        _mode[0] = new_mode
        log(f"Mode → {new_mode}")
        ui.send({"t": "mode",   "v": new_mode})
        ui.send({"t": "status", "v": "○  Idle"})
        if new_mode == "STREAMING":
            audio.enable_streaming(True)
            if _mode_item_ref[0]:
                _mode_item_ref[0].title = "Switch to FILE mode"
        else:
            audio.enable_streaming(False)
            if _mode_item_ref[0]:
                _mode_item_ref[0].title = "Switch to STREAMING mode"

    # ── UI command handler ────────────────────────────────────────────────────
    _muted = [True]

    def _mute_toggle() -> None:
        _muted[0] = not _muted[0]
        audio.set_muted(_muted[0])
        ui.send({"t": "muted", "v": _muted[0]})

    def handle_ui_command(cmd: dict) -> None:
        c = cmd.get("cmd")
        if c == "switch_mode":
            new = cmd.get("mode", "FICHIER" if _mode[0] == "STREAMING" else "STREAMING")
            _set_mode(new)
        elif c == "correct_now":
            threading.Thread(target=_trigger_correction, daemon=True).start()
        elif c == "record_toggle":
            threading.Thread(target=_record_toggle,      daemon=True).start()
        elif c == "mute_toggle":
            _mute_toggle()
        elif c == "clear":
            _transcript[0] = ""
            ui.send({"t": "transcript",    "v": ""})
            ui.send({"t": "clear_corrected"})
        elif c == "import_file":
            path = cmd.get("path", "")
            if path:
                threading.Thread(target=_import_file, args=(path,), daemon=True).start()
        elif c == "exit":
            rumps.quit_application()

    _ui_cmd_handler[0] = handle_ui_command   # resolve circular dependency

    # ── Keyboard shortcut listener ────────────────────────────────────────────
    class _HotkeyListener:
        _K = frozenset([keyboard.Key.cmd, keyboard.Key.shift,
                        keyboard.KeyCode.from_char("k")])
        _R = frozenset([keyboard.Key.cmd, keyboard.Key.shift,
                        keyboard.KeyCode.from_char("r")])

        def __init__(self) -> None:
            self._held  : set  = set()
            self._fired : bool = False

        def start(self) -> bool:
            try:
                keyboard.Listener(
                    on_press   = self._press,
                    on_release = self._release,
                ).start()
                log("Keyboard listener active (Cmd+Shift+K / Cmd+Shift+R)", "OK")
                return True
            except Exception as exc:
                log(f"pynput : {exc}", "ERR")
                return False

        @staticmethod
        def _norm(key):
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                return keyboard.Key.shift
            if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                return keyboard.Key.cmd
            return key

        def _press(self, key) -> None:
            self._held.add(self._norm(key))
            if self._fired:
                return
            if self._held >= self._K:
                self._fired = True
                log("Cmd+Shift+K detected")
                threading.Thread(target=_trigger_correction, daemon=True).start()
            elif self._held >= self._R:
                self._fired = True
                log("Cmd+Shift+R detected")
                threading.Thread(target=_record_toggle, daemon=True).start()

        def _release(self, key) -> None:
            self._held.discard(self._norm(key))
            if not self._held:
                self._fired = False

    # ── rumps menu bar app ────────────────────────────────────────────────────
    class DicteeApp(rumps.App):
        def __init__(self) -> None:
            mode_item = rumps.MenuItem(
                "Switch to FILE mode",
                callback=self._on_mode_toggle,
            )
            _mode_item_ref[0] = mode_item
            super().__init__(
                "✍️",
                menu        = [mode_item, None],
                quit_button = "Quit",
            )
            self._boot()

        def _boot(self) -> None:
            # 1. Start UI
            ui.ensure_alive()
            ui.send({"t": "mode",  "v": "STREAMING"})
            audio.set_muted(True)
            ui.send({"t": "muted", "v": True})

            # 2. Keyboard shortcuts
            ok = _HotkeyListener().start()
            if not ok:
                ui.send({"t": "status",
                         "v": "⚠  Accessibility permission missing — System Settings → Accessibility"})
                rumps.notification(
                    "Dictation Corrector",
                    "Accessibility permission missing",
                    "System Settings → Privacy & Security "
                    "→ Accessibility → enable Terminal.",
                    sound=True,
                )

            # 3. Open mic stream + load Parakeet async.
            #    Streaming only enabled once Parakeet is ready (on_ready callback)
            #    → no "chunk dropped" messages during loading.
            audio.start_stream()
            parakeet.load_async(on_ready=lambda: audio.enable_streaming(True))

        def _on_mode_toggle(self, sender) -> None:
            new = "FICHIER" if _mode[0] == "STREAMING" else "STREAMING"
            _set_mode(new)

    log("Dictation Corrector v2 started.")
    log("Cmd+Shift+K: correct  |  Cmd+Shift+R: record (FILE mode)")
    log(f"Ollama : {OLLAMA_URL}  modèle : {OLLAMA_MODEL}")
    log(f"Parakeet : {PARAKEET_MODEL}")
    DicteeApp().run()


# ─────────────────────────────────────────────────────────────────────────────
# Startup environment check
# ─────────────────────────────────────────────────────────────────────────────

def _preflight() -> bool:
    """
    Checks the environment before startup.
    Prints a clear summary to stderr.
    Returns False if a blocking prerequisite is missing.
    """
    import importlib, os
    ok_all = True

    def chk(label: str, ok: bool, detail: str = "", blocking: bool = True) -> bool:
        nonlocal ok_all
        sym  = "✓" if ok else ("✗" if blocking else "⚠")
        info = f"  — {detail}" if detail else ""
        print(f"  {sym}  {label}{info}", file=sys.stderr, flush=True)
        if not ok and blocking:
            ok_all = False
        return ok

    sep = "─" * 52
    print(sep, file=sys.stderr, flush=True)
    print("  Environment check", file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)

    # ── Python ────────────────────────────────────────────────────────────────
    v = sys.version_info
    chk(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 11),
        detail="" if v >= (3, 11) else "Python 3.11+ required")

    # ── Python dependencies ───────────────────────────────────────────────────
    for pkg, import_name in [
        ("nemo_toolkit", "nemo"),
        ("sounddevice",  "sounddevice"),
        ("soundfile",    "soundfile"),
        ("httpx",        "httpx"),
        ("pyperclip",    "pyperclip"),
        ("pynput",       "pynput"),
        ("rumps",        "rumps"),
    ]:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            chk(pkg, True, ver)
        except ImportError:
            chk(pkg, False, f"pip install {pkg}")

    # ── PyTorch + MPS ─────────────────────────────────────────────────────────
    try:
        import torch
        mps = torch.backends.mps.is_available()
        chk(f"PyTorch {torch.__version__}", True,
            "MPS (Apple Silicon)" if mps else "CPU only (MPS unavailable)",
            blocking=False)
    except ImportError:
        chk("PyTorch", False, "installed with nemo_toolkit")

    # ── Parakeet model cache ──────────────────────────────────────────────────
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--nvidia--parakeet-tdt-0.6b-v3",
    )
    cached = os.path.isdir(cache_dir)
    chk(f"Parakeet  {PARAKEET_MODEL}",
        cached,
        "cached" if cached else "will be downloaded on first run (~600 MB)",
        blocking=False)

    # ── Microphone (sounddevice) ──────────────────────────────────────────────
    try:
        import sounddevice as sd
        devices    = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        default    = sd.query_devices(kind="input")
        chk("Microphone", True, f"{default['name']!r}  ({len(input_devs)} input(s) found)")
    except Exception as exc:
        chk("Microphone", False, str(exc)[:70])

    # ── Ollama ────────────────────────────────────────────────────────────────
    try:
        resp = httpx.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=3.0)
        resp.raise_for_status()
        models_available = [m["name"] for m in resp.json().get("models", [])]
        chk("Ollama", True, OLLAMA_URL)
        model_present = any(
            m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL.split(":")[0])
            for m in models_available
        )
        chk(f"Model  {OLLAMA_MODEL}", model_present,
            "available" if model_present else
            f"not found — run: ollama pull {OLLAMA_MODEL}")
    except httpx.ConnectError:
        chk("Ollama", False, f"unreachable at {OLLAMA_URL} — run: ollama serve")
        chk(f"Model  {OLLAMA_MODEL}", False, "Ollama not available")
    except Exception as exc:
        chk("Ollama", False, str(exc)[:70])

    print(sep, file=sys.stderr, flush=True)
    if ok_all:
        print("  ✓  All good — starting.", file=sys.stderr, flush=True)
    else:
        print("  ✗  Blocking prerequisites missing (see ✗ above).", file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)

    return ok_all


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        _run_ui_mode()
    else:
        _preflight()
        _run_daemon_mode()
