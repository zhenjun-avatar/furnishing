"""进程内缓存：模拟端上传的图片（替代 Dify /files/upload）。"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CachedUpload:
    data: bytes
    mime: str


_TTL_SECONDS = 3600.0
_MAX_KEYS = 512
_PERSIST_DIR = Path(__file__).resolve().parents[2] / "outputs" / "simulate_uploads"


class SimulateUploadStore:
    """upload_file_id → 字节；供 LangGraph 多模态消息构造 data URI。"""

    _singleton: SimulateUploadStore | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._data: dict[str, tuple[CachedUpload, float]] = {}
        self._lock = threading.Lock()
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> SimulateUploadStore:
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = SimulateUploadStore()
            return cls._singleton

    def put(self, data: bytes, mime: str) -> str:
        token = secrets.token_urlsafe(18)
        now = time.monotonic()
        normalized_mime = (mime or "image/jpeg").strip() or "image/jpeg"
        with self._lock:
            self._purge_unlocked(now)
            self._data[token] = (CachedUpload(data=data, mime=normalized_mime), now)
            self._persist_unlocked(token, data, normalized_mime)
        return token

    def get(self, token: str) -> CachedUpload | None:
        if not token:
            return None
        now = time.monotonic()
        with self._lock:
            self._purge_unlocked(now)
            hit = self._data.get(token)
            if hit is None:
                loaded = self._load_persisted_unlocked(token)
                if loaded is None:
                    return None
                self._data[token] = (loaded, now)
                return loaded
            entry, _ = hit
            self._data[token] = (entry, now)
            return entry

    def _persist_unlocked(self, token: str, data: bytes, mime: str) -> None:
        (_PERSIST_DIR / f"{token}.bin").write_bytes(data)
        (_PERSIST_DIR / f"{token}.mime").write_text(mime, encoding="utf-8")

    def _load_persisted_unlocked(self, token: str) -> CachedUpload | None:
        blob = _PERSIST_DIR / f"{token}.bin"
        if not blob.is_file():
            return None
        mime_path = _PERSIST_DIR / f"{token}.mime"
        mime = "image/jpeg"
        if mime_path.is_file():
            try:
                mime = (mime_path.read_text(encoding="utf-8") or "").strip() or mime
            except Exception:
                pass
        try:
            data = blob.read_bytes()
            return CachedUpload(data=data, mime=mime)
        except Exception:
            return None

    def _purge_unlocked(self, now: float) -> None:
        dead = [k for k, (_, ts) in self._data.items() if now - ts > _TTL_SECONDS]
        for k in dead:
            del self._data[k]
        while len(self._data) > _MAX_KEYS:
            oldest = min(self._data.items(), key=lambda kv: kv[1][1])[0]
            del self._data[oldest]
