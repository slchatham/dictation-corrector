#!/usr/bin/env python3
"""
Dictée Correcteur v2
Pipeline : Micro → Parakeet ASR (nvidia/parakeet-tdt-0.6b-v3) → Qwen3.5:4b → fenêtre flottante

Modes audio :
  STREAMING  capture continue par chunks de 2s, Cmd+Shift+K déclenche la correction
  FICHIER    Cmd+Shift+R start/stop, transcription batch, correction automatique

Raccourcis :
  Cmd+Shift+K  correction LLM sur le buffer courant
  Cmd+Shift+R  start/stop enregistrement (mode FICHIER)

Dépendances :
  pip install nemo_toolkit[asr] sounddevice soundfile numpy httpx pynput rumps

Lancement :
  python dictee_correcteur.py
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
CHUNK_SECONDS  = 2

SYSTEM_PROMPT = """\
Tu es un correcteur de dictée vocale pour un auteur bilingue français/anglais.
Le texte t'est envoyé tel que capturé par un ASR. L'auteur alterne librement entre les deux langues.

Corrige UNIQUEMENT ces trois cas :
1. Mot anglais phonétisé à la française par l'ASR  →  rétablis l'orthographe anglaise
   ex : "à l'échelle" → "at scale", "dîtes" → "data", "baguette" → "backlog"
2. Erreur de transcription phonétique évidente dans la langue en cours
   ex : "pansant" → "pensant",  "their" → "there"
3. Mot de ponctuation dicté oralement
   ex : "virgule" → ",",  "point" → ".",  "à la ligne" → "\\n",  "tiret" → "–"

NE JAMAIS :
- traduire une phrase d'une langue à l'autre (l'anglais intentionnel reste anglais, le français reste français)
- reformuler, compléter ou corriger le fond
- modifier la ponctuation existante

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
# Filtre <think>…</think>  (tokens de réflexion qwen3)
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
# MODE UI  ──  python dictee_correcteur.py --ui
# Fenêtre tkinter persistante.
# Reçoit des mises à jour du daemon via stdin (JSON-lines).
# Envoie des commandes au daemon via stdout (JSON-lines).
# ─────────────────────────────────────────────────────────────────────────────

def _run_ui_mode() -> None:
    import tkinter as tk

    log("Mode UI démarré")

    # ── Palette Catppuccin Mocha ──────────────────────────────────────────────
    C = dict(
        bg      = "#1e1e2e",
        surface = "#313244",
        overlay = "#45475a",
        text    = "#cdd6f4",
        sub     = "#a6adc8",
        dim     = "#585b70",
        blue    = "#89b4fa",
        green   = "#a6e3a1",
        red     = "#f38ba8",
        yellow  = "#f9e2af",
        teal    = "#94e2d5",
    )

    # ── Envoyer une commande au daemon via stdout ─────────────────────────────
    def send_cmd(cmd: dict) -> None:
        try:
            sys.stdout.write(json.dumps(cmd) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            log(f"send_cmd: {exc}", "ERR")

    # ── Root ─────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Dictée Correcteur v2")
    root.configure(bg=C["bg"])
    root.attributes("-topmost", True)
    root.geometry("700x620")
    root.resizable(True, True)

    # ── Variables dynamiques ──────────────────────────────────────────────────
    mode_var   = tk.StringVar(value="STREAMING")
    status_var = tk.StringVar(value="⟳  Chargement Parakeet…")
    rec_var    = tk.StringVar(value="⏺  Enregistrer")

    # ── Header ───────────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg=C["bg"])
    hdr.pack(fill="x", padx=14, pady=(14, 6))

    tk.Label(hdr, text="Dictée Correcteur v2",
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
    tk.Label(hdr, text="Mode :", bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 10)).pack(side="right", padx=(0, 4))

    # ── Transcription brute Parakeet ──────────────────────────────────────────
    tk.Label(root, text="  TRANSCRIPTION PARAKEET",
             bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 9)).pack(anchor="w", padx=14, pady=(4, 2))

    f_raw = tk.Frame(root, bg=C["surface"])
    f_raw.pack(fill="x", padx=12, pady=(0, 8))

    w_raw = tk.Text(
        f_raw, wrap="word", bg=C["surface"], fg=C["dim"],
        font=("Helvetica Neue", 12), bd=0, relief="flat",
        height=5, padx=10, pady=8, state="disabled",
    )
    w_raw.pack(fill="both")

    # ── Correction Qwen3.5 ───────────────────────────────────────────────────
    tk.Label(root, text="  CORRECTION QWEN3.5",
             bg=C["bg"], fg=C["dim"],
             font=("Helvetica Neue", 9)).pack(anchor="w", padx=14, pady=(0, 2))

    f_corr = tk.Frame(root, bg=C["surface"])
    f_corr.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    w_corr = tk.Text(
        f_corr, wrap="word", bg=C["surface"], fg=C["text"],
        font=("Helvetica Neue", 13), bd=0, relief="flat",
        height=9, padx=10, pady=8, state="disabled",
    )
    w_corr.pack(fill="both", expand=True)

    # ── Barre de statut ───────────────────────────────────────────────────────
    tk.Label(root, textvariable=status_var,
             bg=C["bg"], fg=C["blue"],
             font=("Helvetica Neue", 11)).pack(pady=(2, 4))

    # ── Boutons ───────────────────────────────────────────────────────────────
    bf = tk.Frame(root, bg=C["bg"])
    bf.pack(fill="x", padx=12, pady=(0, 14))

    # Bouton enregistrement — visible uniquement en mode FICHIER
    rec_btn = tk.Button(
        bf, textvariable=rec_var,
        bg=C["surface"], fg=C["text"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "record_toggle"}),
    )

    tk.Button(
        bf, text="Corriger maintenant",
        bg=C["teal"], fg=C["bg"], activebackground=C["sub"],
        font=("Helvetica Neue", 11, "bold"), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "correct_now"}),
    ).pack(side="left", padx=(0, 8))

    mute_var = tk.StringVar(value="🎙 Actif")
    mute_btn = tk.Button(
        bf, textvariable=mute_var,
        bg=C["surface"], fg=C["green"], activebackground=C["overlay"],
        font=("Helvetica Neue", 11), bd=0, relief="flat", padx=12, pady=7,
        command=lambda: send_cmd({"cmd": "mute_toggle"}),
    )
    mute_btn.pack(side="left", padx=(0, 8))

    def copy_corrected() -> None:
        w_corr.config(state="normal")
        text = w_corr.get("1.0", "end").strip()
        w_corr.config(state="disabled")
        if text:
            pyperclip.copy(text)
            copy_btn.config(text="✓  Copié !")
            root.after(2000, lambda: copy_btn.config(text="Copier"))

    copy_btn = tk.Button(
        bf, text="Copier",
        bg=C["blue"], fg=C["bg"], activebackground=C["sub"],
        font=("Helvetica Neue", 11, "bold"), bd=0, relief="flat", padx=12, pady=7,
        command=copy_corrected,
    )
    copy_btn.pack(side="right")

    # ── Helpers widgets (thread principal) ────────────────────────────────────
    def _set_widget(w: tk.Text, text: str) -> None:
        w.config(state="normal")
        w.delete("1.0", "end")
        w.insert("1.0", text)
        w.config(state="disabled")

    def _append_widget(w: tk.Text, text: str) -> None:
        w.config(state="normal")
        w.insert("end", text)
        w.config(state="disabled")
        w.see("end")

    # ── Gestionnaire de messages entrants ─────────────────────────────────────
    def handle_msg(msg: dict) -> None:
        t = msg.get("t")
        v = msg.get("v")
        if t == "status":
            status_var.set(v)
        elif t == "mode":
            mode_var.set(v)
            if v == "FICHIER":
                rec_btn.pack(side="left", padx=(0, 8), before=copy_btn)
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
                rec_btn.config(bg=C["red"], fg=C["bg"])
            else:
                rec_var.set("⏺  Enregistrer")
                rec_btn.config(bg=C["surface"], fg=C["text"])
        elif t == "muted":
            if v:
                mute_var.set("🔇 Muet")
                mute_btn.config(bg=C["overlay"], fg=C["dim"])
            else:
                mute_var.set("🎙 Actif")
                mute_btn.config(bg=C["surface"], fg=C["green"])

    # ── Lecteur stdin (thread d'arrière-plan) ─────────────────────────────────
    def read_stdin() -> None:
        import io
        reader = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
        for raw in reader:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
                root.after(0, lambda m=msg: handle_msg(m))
            except Exception as exc:
                log(f"stdin parse: {exc}", "ERR")
        log("Stdin fermé (daemon arrêté), fermeture UI")
        root.after(0, root.destroy)

    threading.Thread(target=read_stdin, daemon=True).start()

    # ── Centrage ──────────────────────────────────────────────────────────────
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h   = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")

    log("Fenêtre tkinter ouverte")
    root.mainloop()
    log("Fenêtre tkinter fermée")


# ─────────────────────────────────────────────────────────────────────────────
# MODE DAEMON  ──  python dictee_correcteur.py  (défaut)
# ─────────────────────────────────────────────────────────────────────────────

def _run_daemon_mode() -> None:
    import subprocess
    import rumps
    from pynput import keyboard
    import sounddevice as sd
    import atexit

    # ── État partagé (listes mutables pour closures) ──────────────────────────
    _mode      = ["STREAMING"]   # "STREAMING" | "FICHIER"
    _recording = [False]
    _transcript = [""]           # buffer de transcription accumulé

    # ── UIBridge — IPC bidirectionnel avec le sous-processus tkinter ──────────
    _ui_cmd_handler = [None]     # défini plus tard, évite la dépendance circulaire

    class _UIBridge:
        def __init__(self) -> None:
            self._proc: subprocess.Popen | None = None
            self._lock = threading.Lock()
            atexit.register(self._cleanup)

        def _cleanup(self) -> None:
            with self._lock:
                if self._proc and self._proc.poll() is None:
                    log("Fermeture sous-processus UI…")
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2)
                    except Exception:
                        self._proc.kill()

        def ensure_alive(self) -> None:
            with self._lock:
                if self._proc is None or self._proc.poll() is not None:
                    log("Lancement sous-processus UI…")
                    self._proc = subprocess.Popen(
                        [sys.executable, __file__, "--ui"],
                        stdin  = subprocess.PIPE,
                        stdout = subprocess.PIPE,
                        stderr = sys.stderr,
                    )
                    log(f"UI PID {self._proc.pid}", "OK")
                    threading.Thread(
                        target = self._read_commands,
                        args   = (self._proc,),
                        daemon = True,
                    ).start()

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
            for raw in reader:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    cmd = json.loads(raw)
                    log(f"Commande UI : {cmd}")
                    handler = _ui_cmd_handler[0]
                    if handler:
                        handler(cmd)
                except Exception as exc:
                    log(f"UI cmd parse: {exc}", "ERR")

    ui = _UIBridge()

    # ── Moteur Parakeet ───────────────────────────────────────────────────────
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
            NeMo réinitialise son logging à chaque opération (from_pretrained, transcribe…).
            On balaie TOUS les loggers enregistrés et on force ERROR sur ceux de NeMo/Lightning.
            À appeler après le chargement du modèle ET avant chaque transcription.
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
            warnings.filterwarnings("ignore")   # warnings PyTorch/NeMo à l'init
            self._silence_nemo_loggers()        # passe initiale avant import NeMo
            try:
                import torch
                import nemo.collections.asr as nemo_asr

                log(f"Chargement {PARAKEET_MODEL}…")
                ui.send({"t": "status", "v": "⟳  Chargement Parakeet…"})

                self._model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL)

                if torch.backends.mps.is_available():
                    try:
                        self._model = self._model.to("mps")
                        self._device = "mps"
                        log("Parakeet sur MPS (Apple Silicon)", "OK")
                    except Exception as exc:
                        log(f"MPS indisponible ({exc}), fallback CPU", "WARN")
                        self._device = "cpu"
                else:
                    log("MPS non disponible, Parakeet sur CPU")

                self._model.eval()
                self._silence_nemo_loggers()    # re-passe après from_pretrained
                log(f"Parakeet prêt sur {self._device}", "OK")
                ui.send({"t": "status", "v": "○  Idle"})
                self.ready.set()
                if on_ready:
                    on_ready()

            except Exception as exc:
                log(f"Erreur chargement Parakeet : {exc}", "ERR")
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
                    self._silence_nemo_loggers()    # NeMo recrée des loggers à chaque appel
                    result = self._model.transcribe([fname], verbose=False)
                    # NeMo peut retourner list[str] ou tuple(list[str], ...)
                    if isinstance(result, tuple):
                        result = result[0]
                    text = result[0] if result else ""
                    # Certaines versions retournent un objet Hypothesis
                    if hasattr(text, "text"):
                        text = text.text
                    return str(text).strip()
                except Exception as exc:
                    log(f"transcribe() erreur : {exc}", "ERR")
                    return ""
                finally:
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass

    parakeet = _ParakeetEngine()

    # ── Moteur audio (sounddevice) ────────────────────────────────────────────
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
                    blocksize  = int(SAMPLE_RATE * 0.1),   # blocs 100 ms
                    callback   = self._callback,
                )
                self._stream.start()
                log("Stream audio ouvert", "OK")
            except Exception as exc:
                log(f"sounddevice : {exc}", "ERR")
                if "permission" in str(exc).lower() or "access" in str(exc).lower():
                    ui.send({"t": "status",
                             "v": "✗  Permission micro refusée — autorisez dans Préférences Système"})
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
            log(f"Micro {'coupé' if muted else 'actif'}", "OK")

        def _callback(self, indata, frames, time, status) -> None:
            if status:
                log(f"Audio status : {status}", "WARN")
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

    # ── Worker de transcription (thread dédié) ────────────────────────────────
    def _transcription_worker() -> None:
        log("Worker transcription démarré")
        while True:
            chunk = _transcription_queue.get()
            if chunk is None:
                break
            if not parakeet.ready.is_set():
                log("Parakeet pas encore prêt, chunk ignoré", "WARN")
                continue
            dur = len(chunk) / SAMPLE_RATE
            log(f"Transcription chunk {dur:.1f}s…")
            ui.send({"t": "status", "v": "◎  Transcription…"})
            text = parakeet.transcribe(chunk)
            if text:
                sep = " " if _transcript[0] else ""
                _transcript[0] += sep + text
                log(f"→ '{text[:80]}'")
                ui.send({"t": "transcript", "v": _transcript[0]})
            ui.send({"t": "status", "v": "○  Idle"})

    threading.Thread(target=_transcription_worker, daemon=True).start()

    # ── Correction LLM ────────────────────────────────────────────────────────
    _correction_lock = threading.Lock()

    def _trigger_correction(text: str | None = None) -> None:
        text = text or _transcript[0]
        if not text.strip():
            log("Buffer vide, correction ignorée", "WARN")
            ui.send({"t": "status", "v": "⚠  Rien à corriger (buffer vide)"})
            return
        threading.Thread(target=_run_correction, args=(text,), daemon=True).start()

    def _run_correction(text: str) -> None:
        if not _correction_lock.acquire(blocking=False):
            log("Correction déjà en cours", "WARN")
            return
        try:
            log(f"Correction LLM ({len(text)} car.)…")
            ui.send({"t": "status", "v": "✦  Correction en cours…"})
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
                            log(f"Correction terminée — {tokens} tokens en {dur:.1f}s", "OK")
                            break
                    tail = flt.flush()
                    if tail:
                        ui.send({"t": "corrected_chunk", "v": tail})
            ui.send({"t": "status", "v": "✓  Correction terminée"})
        except httpx.ConnectError:
            log(f"Ollama inaccessible sur {OLLAMA_URL}", "ERR")
            ui.send({"t": "status", "v": "✗  Ollama inaccessible — lancez : ollama serve"})
        except Exception as exc:
            log(f"Erreur correction : {exc}", "ERR")
            ui.send({"t": "status", "v": f"✗  {str(exc)[:80]}"})
        finally:
            _correction_lock.release()

    # ── Toggle enregistrement (mode FICHIER) ──────────────────────────────────
    def _record_toggle() -> None:
        if _mode[0] != "FICHIER":
            log("Record ignoré hors mode FICHIER", "WARN")
            return
        if not _recording[0]:
            log("Démarrage enregistrement…")
            _transcript[0] = ""
            ui.send({"t": "transcript", "v": ""})
            ui.send({"t": "clear_corrected"})
            audio.start_recording()
            ui.send({"t": "recording", "v": True})
            ui.send({"t": "status",    "v": "⏺  Enregistrement…"})
        else:
            log("Arrêt enregistrement…")
            raw_audio = audio.stop_recording()
            ui.send({"t": "recording", "v": False})
            if len(raw_audio) < SAMPLE_RATE * 0.5:
                log("Enregistrement trop court (<0.5s)", "WARN")
                ui.send({"t": "status", "v": "⚠  Enregistrement trop court"})
                return
            dur = len(raw_audio) / SAMPLE_RATE
            log(f"Transcription batch {dur:.1f}s…")
            ui.send({"t": "status", "v": f"◎  Transcription batch ({dur:.1f}s)…"})

            def _batch() -> None:
                text = parakeet.transcribe(raw_audio)
                if text:
                    _transcript[0] = text
                    log(f"Batch → '{text[:80]}'", "OK")
                    ui.send({"t": "transcript", "v": text})
                    _trigger_correction(text)
                else:
                    log("Transcription batch vide", "WARN")
                    ui.send({"t": "status", "v": "⚠  Transcription vide"})

            threading.Thread(target=_batch, daemon=True).start()

    # ── Switch de mode ────────────────────────────────────────────────────────
    _mode_item_ref = [None]   # référence au MenuItem rumps, défini plus tard

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
                _mode_item_ref[0].title = "Passer en mode FICHIER"
        else:
            audio.enable_streaming(False)
            if _mode_item_ref[0]:
                _mode_item_ref[0].title = "Passer en mode STREAMING"

    # ── Gestionnaire des commandes venant de l'UI ─────────────────────────────
    _muted = [False]

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

    _ui_cmd_handler[0] = handle_ui_command   # résoudre la dépendance circulaire

    # ── Listener raccourcis clavier ───────────────────────────────────────────
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
                log("Listener clavier actif (Cmd+Shift+K / Cmd+Shift+R)", "OK")
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
                log("Cmd+Shift+K détecté")
                threading.Thread(target=_trigger_correction, daemon=True).start()
            elif self._held >= self._R:
                self._fired = True
                log("Cmd+Shift+R détecté")
                threading.Thread(target=_record_toggle, daemon=True).start()

        def _release(self, key) -> None:
            self._held.discard(self._norm(key))
            if not self._held:
                self._fired = False

    # ── App rumps (menu bar macOS) ────────────────────────────────────────────
    class DicteeApp(rumps.App):
        def __init__(self) -> None:
            mode_item = rumps.MenuItem(
                "Passer en mode FICHIER",
                callback=self._on_mode_toggle,
            )
            _mode_item_ref[0] = mode_item
            super().__init__(
                "✍️",
                menu        = [mode_item, None],
                quit_button = "Quitter",
            )
            self._boot()

        def _boot(self) -> None:
            # 1. Démarrer l'UI
            ui.ensure_alive()
            ui.send({"t": "mode", "v": "STREAMING"})

            # 2. Raccourcis clavier
            ok = _HotkeyListener().start()
            if not ok:
                ui.send({"t": "status",
                         "v": "⚠  Accessibilité manquante — Préférences Système → Accessibilité"})
                rumps.notification(
                    "Dictée Correcteur",
                    "Permission Accessibilité manquante",
                    "Préférences Système → Confidentialité & Sécurité "
                    "→ Accessibilité → activez Terminal.",
                    sound=True,
                )

            # 3. Ouvrir le flux micro + charger Parakeet async.
            #    Le streaming n'est activé qu'une fois Parakeet prêt (callback on_ready)
            #    → plus de messages "chunk ignoré" pendant le chargement.
            audio.start_stream()
            parakeet.load_async(on_ready=lambda: audio.enable_streaming(True))

        def _on_mode_toggle(self, sender) -> None:
            new = "FICHIER" if _mode[0] == "STREAMING" else "STREAMING"
            _set_mode(new)

    log("Dictée Correcteur v2 démarré.")
    log("Cmd+Shift+K : correction  |  Cmd+Shift+R : enregistrement (mode FICHIER)")
    log(f"Ollama : {OLLAMA_URL}  modèle : {OLLAMA_MODEL}")
    log(f"Parakeet : {PARAKEET_MODEL}")
    DicteeApp().run()


# ─────────────────────────────────────────────────────────────────────────────
# Vérification au démarrage
# ─────────────────────────────────────────────────────────────────────────────

def _preflight() -> bool:
    """
    Vérifie l'environnement avant le démarrage.
    Affiche un résumé clair dans les logs.
    Retourne False si un prérequis bloquant est absent.
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
    print("  Vérification de l'environnement", file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)

    # ── Python ────────────────────────────────────────────────────────────────
    v = sys.version_info
    chk(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 11),
        detail="" if v >= (3, 11) else "Python 3.11+ requis")

    # ── Dépendances Python ────────────────────────────────────────────────────
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
            "MPS (Apple Silicon)" if mps else "CPU uniquement (MPS indisponible)",
            blocking=False)
    except ImportError:
        chk("PyTorch", False, "installé avec nemo_toolkit")

    # ── Modèle Parakeet en cache ──────────────────────────────────────────────
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--nvidia--parakeet-tdt-0.6b-v3",
    )
    cached = os.path.isdir(cache_dir)
    chk(f"Parakeet  {PARAKEET_MODEL}",
        cached,
        "en cache" if cached else "sera téléchargé au 1er lancement (~600 MB)",
        blocking=False)

    # ── Micro (sounddevice) ───────────────────────────────────────────────────
    try:
        import sounddevice as sd
        devices    = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        default    = sd.query_devices(kind="input")
        chk("Micro", True, f"{default['name']!r}  ({len(input_devs)} entrée(s) détectée(s))")
    except Exception as exc:
        chk("Micro", False, str(exc)[:70])

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
        chk(f"Modèle  {OLLAMA_MODEL}", model_present,
            "disponible" if model_present else
            f"non trouvé — lancez : ollama pull {OLLAMA_MODEL}")
    except httpx.ConnectError:
        chk("Ollama", False, f"inaccessible sur {OLLAMA_URL} — lancez : ollama serve")
        chk(f"Modèle  {OLLAMA_MODEL}", False, "Ollama non disponible")
    except Exception as exc:
        chk("Ollama", False, str(exc)[:70])

    print(sep, file=sys.stderr, flush=True)
    if ok_all:
        print("  ✓  Tout est prêt — démarrage.", file=sys.stderr, flush=True)
    else:
        print("  ✗  Des prérequis bloquants sont manquants (voir ✗ ci-dessus).", file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)

    return ok_all


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        _run_ui_mode()
    else:
        _preflight()
        _run_daemon_mode()
