"""Reach's voice: Chatterbox TTS, through the local Chatterbox API (wave1/chatterbox-api).

Chatterbox sounds natural but takes seconds per line on a laptop GPU, too slow to wait
for mid-conversation. Almost everything Reach says comes from a small set of templates
times the items it knows (lines.py), so those lines are rendered ahead of time in the
background and kept on disk. At speaking time a line is either ready (played at once) or
moved to the front of the queue for next time, and the page uses the browser's voice for
that one line.

One worker, one request in flight, bounded queues, and backoff when Chatterbox is down:
Reach must never pile work onto a 16 GB Mac.

    CHATTERBOX_URL   default http://127.0.0.1:8004   (POST /v1/audio/speech, OpenAI request shape)
    REACH_VOICE      0 turns the voice off (the page then always uses the browser's voice)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import struct
import time
import wave
from collections import deque
from pathlib import Path

import httpx

log = logging.getLogger("reach.tts")

MAX_CHARS = 300
URGENT_MAX = 8
LATER_MAX = 300
MIN_SECONDS = 0.2
BACKOFF_FIRST, BACKOFF_MAX = 5.0, 300.0
CACHE_BYTES, CACHE_FILES = 200 * 1024 * 1024, 2000


def key(text: str) -> str:
    return " ".join(str(text).split())


def wav_seconds(data: bytes) -> float:
    """Length of the audio actually present in a WAV, in seconds (0 if it isn't one): the header's
    frame count isn't trusted, so a truncated file measures short. The stdlib reads PCM; Chatterbox
    may write float samples, which wave refuses, so those are measured from the chunk headers."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            frame = w.getsampwidth() * w.getnchannels()
            if not frame or not w.getframerate():
                return 0.0
            return len(w.readframes(w.getnframes())) // frame / w.getframerate()
    except (wave.Error, EOFError):
        pass
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0.0
    i, byte_rate, size = 12, 0, 0
    try:
        while i + 8 <= len(data):
            cid, n = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
            if cid == b"fmt " and n >= 16:
                byte_rate = struct.unpack("<I", data[i + 16:i + 20])[0]
            elif cid == b"data":
                size = min(n, len(data) - i - 8)  # the bytes really in the buffer, not the header's claim
                break
            i += 8 + n + (n & 1)
    except struct.error:  # truncated header
        return 0.0
    return size / byte_rate if byte_rate else 0.0


class Voice:
    def __init__(self, cache_dir: Path, transport: httpx.AsyncBaseTransport | None = None):
        self.url = os.getenv("CHATTERBOX_URL", "http://127.0.0.1:8004").rstrip("/")
        self.enabled = os.getenv("REACH_VOICE", "1") != "0"
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transport = transport
        self.known: set[str] = set()  # the catalog: the only lines the page may ask for
        self.order: list[str] = []  # the catalog, most important first (refills `later` past LATER_MAX)
        self.refused: set[str] = set()  # lines Chatterbox rejected; never asked for again
        self.urgent: deque[str] = deque()
        self.later: deque[str] = deque()
        self.queued: set[str] = set()  # in urgent, in later, or in flight
        self.current: str | None = None  # in flight
        self.bypass = False  # a new urgent line may go out during backoff
        self.wake = asyncio.Event()
        self.worker: asyncio.Task | None = None
        self.failures = 0
        self.rendered = 0
        self.retry_at = 0.0

    def path(self, text: str) -> Path:
        return self.dir / (hashlib.sha256(key(text).encode()).hexdigest()[:24] + ".wav")

    def ready(self, text: str) -> Path | None:
        p = self.path(text)
        try:
            if p.stat().st_size >= 1024:
                with p.open("rb") as f:
                    head = f.read(12)
                if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
                    return p
        except FileNotFoundError:
            return None
        p.unlink(missing_ok=True)  # truncated or not a WAV: render it again
        return None

    def catalog(self, lines) -> None:
        """Replace the known lines and queue the ones not on disk yet, in the order given."""
        lines = list(dict.fromkeys(k for k in map(key, lines) if k and len(k) <= MAX_CHARS))
        self.known, self.order = set(lines), lines
        for t in [t for t in self.later if t not in self.known]:
            self.later.remove(t)
            self.queued.discard(t)
        for t in lines:
            self.want(t)

    def want(self, text: str, urgent: bool = False) -> None:
        """Render this line in the background if it isn't on disk yet. urgent: before everything else."""
        text = key(text)
        if (not self.enabled or not text or len(text) > MAX_CHARS or text == self.current or text in self.refused
                or self.ready(text)):
            return
        if text in self.queued:
            if not (urgent and text in self.later):
                return
            self.later.remove(text)  # promote
        elif not urgent:
            if len(self.later) >= LATER_MAX:
                return
            self.queued.add(text)
            self.later.append(text)
            self.wake.set()
            return
        self.queued.add(text)
        self.urgent.append(text)
        self.bypass = True
        self._trim_urgent()
        self.wake.set()

    def _trim_urgent(self, newest: bool = False) -> None:
        """Evict down to URGENT_MAX: the oldest request, or the newest (to keep a failed line at the head)."""
        while len(self.urgent) > URGENT_MAX:
            old = self.urgent.pop() if newest else self.urgent.popleft()
            if old in self.known and len(self.later) < LATER_MAX:
                self.later.appendleft(old)
            else:
                self.queued.discard(old)

    def start(self) -> None:
        """Start the worker (inside the running event loop). It runs until the server stops; if it ever
        dies, the next start() brings it back. A no-op while it runs."""
        if self.enabled and (self.worker is None or self.worker.done()):
            self.worker = asyncio.create_task(self._work())
            self.worker.add_done_callback(self._ended)

    @staticmethod
    def _ended(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.error("voice worker died; it restarts on the next request", exc_info=task.exception())

    def _refill(self) -> int:
        """Top `later` up from the catalog, in catalog order, with lines not on disk and not queued."""
        added = 0
        for t in self.order:
            if len(self.later) >= LATER_MAX:
                break
            if t not in self.queued and t not in self.refused and t != self.current and not self.ready(t):
                self.queued.add(t)
                self.later.append(t)
                added += 1
        return added

    def status(self) -> dict:
        return {"url": self.url, "enabled": self.enabled, "known": len(self.known),
                "ready": sum(self.path(t).exists() for t in self.known),
                "queued_urgent": len(self.urgent), "queued_later": len(self.later), "rendering": self.current,
                "failures": self.failures, "retry_in": round(max(0.0, self.retry_at - time.monotonic()), 1),
                "rendered": self.rendered}

    def _next(self) -> tuple[str, deque] | None:
        backing_off = time.monotonic() < self.retry_at
        if self.urgent and (not backing_off or self.bypass):
            self.bypass = False
            return self.urgent.popleft(), self.urgent
        if self.later and not backing_off:
            return self.later.popleft(), self.later
        return None

    async def _work(self) -> None:
        timeout = httpx.Timeout(600.0, connect=5.0)  # Chatterbox renders one line at a time; wait our turn
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as c:
            while True:
                nxt = self._next()
                if nxt is None:
                    if not self.urgent and not self.later and self._refill():
                        continue
                    self.wake.clear()
                    wait = max(0.0, self.retry_at - time.monotonic()) if (self.urgent or self.later) else None
                    try:
                        await asyncio.wait_for(self.wake.wait(), wait)
                    except asyncio.TimeoutError:
                        pass
                    continue
                text, q = nxt
                if self.ready(text):
                    self.queued.discard(text)
                    continue
                self.current = text
                try:
                    await self._render(c, text)
                    self.queued.discard(text)
                    self.failures, self.retry_at = 0, 0.0
                    self.rendered += 1
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (400, 413, 422):  # Chatterbox refuses this line; it never will render
                        log.warning("chatterbox refused %r: %s", text, e.response.status_code)
                        self.queued.discard(text)
                        self.refused.add(text)
                    else:
                        self._failed(text, q, e)
                except Exception as e:  # noqa: BLE001 - no voice server: the browser voice is used meanwhile
                    self._failed(text, q, e)
                finally:
                    self.current = None

    def _failed(self, text: str, q: deque, e: Exception) -> None:
        q.appendleft(text)  # back to the head of its queue, still counted as queued
        if q is self.urgent:
            self._trim_urgent(newest=True)
        self.failures += 1
        backoff = min(BACKOFF_MAX, BACKOFF_FIRST * 2 ** min(self.failures - 1, 16))  # no overflow after days down
        self.retry_at = time.monotonic() + backoff
        log.info("chatterbox unavailable (%s); retrying in %.0f s, the browser voice is used meanwhile",
                 e.__class__.__name__ if not str(e) else e, backoff)

    async def _render(self, c: httpx.AsyncClient, text: str) -> None:
        r = await c.post(f"{self.url}/v1/audio/speech", json={"input": text, "response_format": "wav"})
        r.raise_for_status()
        secs = wav_seconds(r.content)
        if secs < MIN_SECONDS:
            raise ValueError(f"not a usable WAV ({len(r.content)} bytes, {secs:.2f} s)")
        dest = self.path(text)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        self._trim_cache()

    def _trim_cache(self) -> None:
        files = []
        for p in self.dir.glob("*.wav"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            files.append((st.st_mtime, st.st_size, p))
        files.sort()
        total = sum(s for _, s, _ in files)
        while files and (total > CACHE_BYTES or len(files) > CACHE_FILES):
            _, size, p = files.pop(0)
            p.unlink(missing_ok=True)
            total -= size
