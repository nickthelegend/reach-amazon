"""Reach's voice with no Chatterbox anywhere: the /api/voice route against a real server process
(whose Chatterbox URL is a dead port), the line catalog against the real fetch stages, and the
render worker against a mock Chatterbox (httpx.MockTransport)."""

import asyncio
import io
import struct
import time
import wave

import httpx
import pytest

from reach import lines, tts
from reach.arm import SimArm
from reach.jobs import Runner
from reach.tts import BACKOFF_MAX, URGENT_MAX, Voice, wav_seconds
from reach.world import World

from test_reach import server  # noqa: F401  (fixture)

PIN = {"X-Reach-Pin": "2468"}


def make_wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return b.getvalue()


# ---------------------------------------------------------------- the route ----
def test_voice_not_ready_returns_204_and_queues_it(server):  # noqa: F811
    h = httpx.Client(base_url=server, timeout=5)
    t0 = time.monotonic()
    r = h.get("/api/voice", params={"text": "Stopped."})
    assert r.status_code == 204 and time.monotonic() - t0 < 0.5  # never waits on Chatterbox
    v = h.get("/api/state").json()["voice"]
    assert v["enabled"] and v["known"] > 50 and v["ready"] == 0
    assert v["failures"] >= 1 or v["queued_urgent"] + v["queued_later"] + (v["rendering"] is not None) > 0
    assert h.get("/api/voice", params={"text": "  Stopped.  "}).status_code == 204  # normalised


def test_voice_serves_cached_wav(server, tmp_path):  # noqa: F811
    data = make_wav()
    Voice(tmp_path / "voice").path("Stopped.").write_bytes(data)  # tmp_path is the server's REACH_DATA
    r = httpx.get(server + "/api/voice", params={"text": "Stopped."}, timeout=5)
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav") and r.content == data


def test_voice_unknown_text_404_and_bad_text_400(server):  # noqa: F811
    h = httpx.Client(base_url=server, timeout=5)
    before = h.get("/api/state").json()["voice"]
    assert h.get("/api/voice", params={"text": "Read me the whole of War and Peace."}).status_code == 404
    after = h.get("/api/state").json()["voice"]
    assert after["queued_urgent"] <= before["queued_urgent"]  # nothing new queued
    assert h.get("/api/voice", params={"text": "   "}).status_code == 400
    assert h.get("/api/voice", params={"text": "x" * 301}).status_code == 400


def test_tool_replies_are_known_lines(server):  # noqa: F811
    h = httpx.Client(base_url=server, timeout=5)
    say = lambda tool, **a: h.post("/api/do", json={"tool": tool, "args": a}, headers=PIN).json()["say"]  # noqa: E731
    replies = [say("list_items"), say("arm_status"), say("got_it"), say("medication_today"), say("fetch_item", item="water"),
               say("fetch_item", item="pills"), say("arm_status")]
    for _ in range(200):
        if h.get("/api/state").json()["job"]["stage"] == "holding it out":
            break
        time.sleep(0.05)
    job = h.get("/api/state").json()["job"]
    assert job["say"] == lines.holding_line("water")
    replies += [job["say"], say("arm_status"), say("got_it")]
    for text in replies + ["Stopped."]:
        assert h.get("/api/voice", params={"text": text}).status_code in (200, 204), text


# ------------------------------------------------------------- the catalog ----
def test_catalog_covers_every_reply_and_stage(tmp_path):
    """Run real fetches (unrouted, and routed on the simulated base) and check that every stage the
    job passes through, and every reply built from it, is a catalog line."""
    async def go():
        world = World(tmp_path / "world.json")
        world.remember("inhaler", [-20, 50, 40, 70, 0, 8], True, [[0.4, 0.8, 0.2]])
        arm = SimArm()
        arm.speedup = 40
        seen: dict[str, list[str]] = {}

        class Recorder(Runner):
            async def _stage(self, job, stage):
                seen.setdefault(job.item, ["starting"]).append(stage)
                await super()._stage(job, stage)

        runner = Recorder(arm, world)
        for name in ("pills", "inhaler"):
            runner.start_fetch(world.places[name])
            for _ in range(400):
                if runner.job.stage == "holding it out":
                    break
                await asyncio.sleep(0.01)
            assert runner.taken()
            await runner.task
        runner.start_return([[0.4, 0.8, 0.2]], "inhaler")
        seen["inhaler"].append(runner.job.stage)
        await runner.task
        return world, seen

    world, seen = asyncio.run(go())
    catalog = set(lines.voice_lines(world.places.values(), has_base=True))
    unrouted = set(lines.voice_lines(world.places.values(), has_base=False))
    assert seen["inhaler"][1] == "going to the inhaler"  # the routed fetch really drove
    for name, stages in seen.items():
        for s in stages:
            assert lines.status_busy(s, name) in catalog, s
            assert lines.got_it_busy(s) in catalog, s
        assert lines.holding_line(name) in catalog
        assert lines.status_done("taken", name) in catalog
    for p in world.places.values():
        assert lines.fetch_reply(p.name, p.route, True) in catalog
        assert lines.fetch_reply(p.name, p.route, False) in unrouted
    assert "about fifteen seconds" not in lines.fetch_reply("inhaler", world.places["inhaler"].route, True)
    assert {lines.STOPPED, lines.LETTING_GO, lines.NOT_HOLDING, lines.RESTING, lines.NO_MEDS,
            lines.can_bring(world.places)} <= catalog
    assert lines.voice_lines(world.places.values(), True)[0] == "Stopped."  # safety first


# -------------------------------------------------------------- the worker ----
async def until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        await asyncio.sleep(0.01)


@pytest.fixture
def voice_env(monkeypatch):
    monkeypatch.delenv("REACH_VOICE", raising=False)
    monkeypatch.setenv("CHATTERBOX_URL", "http://chatterbox.test")


def test_worker_single_flight(tmp_path, voice_env):
    async def go():
        n = {"now": 0, "max": 0, "posts": 0}

        async def handler(req):
            n["now"] += 1
            n["posts"] += 1
            n["max"] = max(n["max"], n["now"])
            await asyncio.sleep(0.01)
            n["now"] -= 1
            return httpx.Response(200, content=make_wav())

        v = Voice(tmp_path, transport=httpx.MockTransport(handler))
        texts = [f"Line number {i}." for i in range(12)]
        v.catalog(texts)
        v.start()
        for t in texts[:5]:
            v.want(t, urgent=True)  # urgent requests mid-batch still go one at a time
        await until(lambda: v.rendered == 12)
        assert n["max"] == 1 and n["posts"] == 12
        assert all(v.ready(t) for t in texts) and not v.queued
        await asyncio.sleep(0.05)
        assert not v.worker.done()  # idle, waiting, never exits
        v.worker.cancel()
    asyncio.run(go())


def test_urgent_dedupe_promotion_and_bound(tmp_path, voice_env):
    async def go():
        v = Voice(tmp_path)  # worker not started: only the bookkeeping
        texts = [f"Line {i}." for i in range(12)]
        v.catalog(texts)
        assert list(v.later) == texts and not v.urgent
        v.want("Line 3.", urgent=True)
        v.want("Line 3.", urgent=True)
        assert list(v.urgent) == ["Line 3."] and "Line 3." not in v.later  # promoted once
        for t in texts:
            v.want(t, urgent=True)
        assert len(v.urgent) == URGENT_MAX
        assert set(v.urgent) | set(v.later) == v.queued == set(texts)  # evicted lines went back to later
        assert not set(v.urgent) & set(v.later)
        v.want("Not in the catalog.", urgent=True)  # want() works for any line; the route is the gate
        assert v.urgent[-1] == "Not in the catalog." and len(v.urgent) == URGENT_MAX
        assert v.queued == set(v.urgent) | set(v.later)
    asyncio.run(go())


def test_failed_line_requeued_backoff_and_urgent_wakes(tmp_path, voice_env):
    async def go():
        mode = {"up": False, "posts": 0}

        def handler(req):
            mode["posts"] += 1
            if not mode["up"]:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, content=make_wav())

        v = Voice(tmp_path, transport=httpx.MockTransport(handler))
        v.catalog(["First.", "Second.", "Third."])
        v.start()
        await until(lambda: v.failures == 1)
        await asyncio.sleep(0.3)
        assert mode["posts"] == 1  # backing off, not hammering
        assert v.later[0] == "First." and "First." in v.queued and v.status()["retry_in"] > 0
        v.want("Second.", urgent=True)  # an urgent line still goes out during backoff...
        await until(lambda: v.failures == 2)
        await asyncio.sleep(0.3)
        assert mode["posts"] == 2 and v.urgent[0] == "Second."  # ...and backs off again when it fails too
        mode["up"] = True
        v.want("Third.", urgent=True)
        await until(lambda: v.ready("Third.") is not None, timeout=2)  # well inside the 10 s backoff
        await until(lambda: v.rendered == 3)  # a success ends the backoff
        assert not v.queued
        v.worker.cancel()
    asyncio.run(go())


def test_invalid_wav_not_cached(tmp_path, voice_env):
    async def go():
        bodies = [b"<html>oops</html>" * 100, make_wav(0.05)]  # not a WAV; a WAV too short to be speech

        def handler(req):
            return httpx.Response(200, content=bodies.pop(0) if bodies else make_wav())

        v = Voice(tmp_path, transport=httpx.MockTransport(handler))
        v.catalog(["Hello."])
        v.start()
        await until(lambda: v.failures == 1)
        assert v.ready("Hello.") is None and not v.path("Hello.").exists() and list(v.later) == ["Hello."]
        v.worker.cancel()
        v.path("Junk.").write_bytes(b"RIFF")  # a truncated file on disk counts as missing and is removed
        assert v.ready("Junk.") is None and not v.path("Junk.").exists()
    asyncio.run(go())
    assert wav_seconds(make_wav(0.05)) < 0.2 <= wav_seconds(make_wav(0.5))


def test_wav_seconds_reads_float_wavs():
    pcm = b"\0" * 32000  # 1 s of float32 mono at 8 kHz
    fmt = struct.pack("<HHIIHH", 3, 1, 8000, 32000, 4, 32)
    data = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt \
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    assert wav_seconds(data) == pytest.approx(1.0)
    assert wav_seconds(b"") == 0 and wav_seconds(b"RIFF....WAVE") == 0
    assert wav_seconds(data[:len(data) - 31200]) == pytest.approx(0.025)  # truncated: what's there, not the header


def test_wav_seconds_counts_frames_present_not_the_header():
    whole = make_wav(0.5)  # 44-byte header claiming 4000 frames
    cut = whole[:44 + 400]  # only 200 frames arrived
    assert wav_seconds(whole) == pytest.approx(0.5)
    assert wav_seconds(cut) == pytest.approx(0.025) and wav_seconds(cut) < tts.MIN_SECONDS


def test_catalog_longer_than_later_max_is_all_rendered_in_order(tmp_path, voice_env, monkeypatch):
    monkeypatch.setattr(tts, "LATER_MAX", 3)

    async def go():
        posted = []

        def handler(req):
            posted.append(req.read().decode())
            return httpx.Response(200, content=make_wav())

        v = Voice(tmp_path, transport=httpx.MockTransport(handler))
        texts = [f"Line {i}." for i in range(10)]
        v.catalog(texts)
        assert len(v.later) == 3  # the rest come from refills when the worker runs dry
        v.start()
        await until(lambda: v.rendered == 10)
        assert [next(t for t in texts if f'"{t}"' in p) for p in posted] == texts and not v.queued
        v.worker.cancel()
    asyncio.run(go())


def test_backoff_never_overflows(tmp_path, voice_env):
    v = Voice(tmp_path)
    v.failures = 5000  # about two weeks of Chatterbox being down
    v._failed("Hello.", v.later, RuntimeError("down"))
    assert 0 < v.status()["retry_in"] <= BACKOFF_MAX and v.later[0] == "Hello."


def test_dead_worker_is_logged_and_restarts(tmp_path, voice_env, caplog):
    async def go():
        v = Voice(tmp_path, transport=httpx.MockTransport(lambda req: httpx.Response(200, content=make_wav())))

        def boom():
            raise RuntimeError("boom")

        v._next = boom
        v.start()
        await until(lambda: v.worker.done())
        await asyncio.sleep(0)  # let the done callback run
        assert "voice worker died" in caplog.text
        del v._next  # back to the real method
        v.catalog(["Hello again."])
        v.start()  # what _voice_catalog() and /api/voice do
        await until(lambda: v.ready("Hello again.") is not None)
        v.worker.cancel()
    asyncio.run(go())
