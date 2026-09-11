"""Tiny JSON-file persistence layer for the brain's state.

Everything the brain knows lives in ``data/*.json`` (git-ignored):
  settings.json  — provider choice + API key (local only, never leaves your machine)
  strategy.json  — the current canonical strategy the indicator implements
  signals.json   — every alert received from TradingView (the learning fuel)
  videos.json    — index of videos that were fed to the brain
  messages.json  — chat history
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _copy(value):
    """Cheap deep copy so callers never mutate the stored default."""
    return json.loads(json.dumps(value))


class JsonStore:
    """Thread-safe JSON file store with atomic-ish writes."""

    def __init__(self, name: str, default):
        self.path = DATA_DIR / name
        self.default = default

    def read(self):
        with _LOCK:
            if not self.path.exists():
                val = self.default() if callable(self.default) else _copy(self.default)
                self._write_unlocked(val)
                return _copy(val)
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                val = self.default() if callable(self.default) else _copy(self.default)
                self._write_unlocked(val)
                return _copy(val)

    def write(self, obj):
        with _LOCK:
            self._write_unlocked(obj)
            return obj

    def _write_unlocked(self, obj):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


class Stores:
    def __init__(self):
        self.settings = JsonStore(
            "settings.json",
            lambda: {
                "provider": "mock",   # mock | gemini | openai
                "api_key": "",
                "model": "",
                "native_video": True,  # Gemini can watch YouTube URLs directly
            },
        )
        self.strategy = JsonStore("strategy.json", lambda: None)
        self.signals = JsonStore("signals.json", lambda: [])
        self.videos = JsonStore("videos.json", lambda: [])
        self.messages = JsonStore("messages.json", lambda: [])

    def add_message(self, role: str, content: str):
        messages = self.messages.read()
        messages.append({"role": role, "content": content, "ts": now_iso()})
        self.messages.write(messages[-200:])

    def add_signal(self, signal: dict) -> dict:
        signals = self.signals.read()
        signals.append(signal)
        self.signals.write(signals[-2000:])
        return signal

    def add_video(self, video: dict) -> dict:
        videos = self.videos.read()
        videos.append(video)
        self.videos.write(videos[-200:])
        return video
