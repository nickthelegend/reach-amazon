"""The Thenar Walker backend against a mock robot speaking the firmware's WebSocket protocol,
route replay on the simulated base, and teach-then-fetch plus the Alexa+ simulator over HTTP."""

import asyncio
import math
import time

import httpx
import numpy as np
import pytest

from reach.arm import SimArm, Stopped
from reach.mock_walker import serve
from reach.walker import WalkerArm

from test_reach import server  # noqa: F401  (fixture)


def run(coro):
    return asyncio.run(coro)


def test_walker_connects_switches_joints_on_one_at_a_time_and_moves():
    async def go():
        robot, srv = await serve(18181)
        try:
            w = WalkerArm("ws://127.0.0.1:18181/")
            await w.connect()
            masks = [c for c in robot.log if c.startswith("M ")]
            assert masks[-1] == "M 63" and len(masks) == 6       # one joint at a time
            assert masks[0] == "M 1" and "S 0" in robot.log       # base first, slow speed indoors
            assert list(w.lo) == [-90, -10, -95, -95, -90, -5]    # limits come from the robot's calibration
            target = w.q.copy()
            target[2] = 30
            await w.move(target)
            assert robot.goal[2] == pytest.approx(30, abs=0.1)
            # driving: the heartbeat keeps the robot's 300 ms failsafe from tripping
            await w.drive(0.5, 0.0, 1.0)
            assert robot.silent_stops == 0
            assert any(c == "D 50 0" for c in robot.log)
            await asyncio.sleep(0.1)  # let the mock robot read what was sent
            assert robot.log[-1].startswith("D 0 0") or "D 0 0" in robot.log[-3:]
            await w.stop()
            await asyncio.sleep(0.1)
            assert "X" in robot.log  # (the latch clears at once: the heartbeat sends a centred stick)
            with pytest.raises(Stopped):
                await w.move(w.q + 5)
        finally:
            srv.close()
    run(go())


def test_walker_refuses_an_uncalibrated_robot():
    async def go():
        robot, srv = await serve(18182)
        robot.cfg = lambda: '{"x":1}' and 'C{"cal":0,"pca":1,"lo":[0,0,0,0,0,0],"hi":[1,1,1,1,1,1],"g":[0,0,0,0,0,0],"usable":0,"on":0,"sp":1}'
        try:
            with pytest.raises(RuntimeError, match="calibration"):
                await WalkerArm("ws://127.0.0.1:18182/").connect()
        finally:
            srv.close()
    run(go())


def test_route_reverse_comes_back_home():
    async def go():
        a = SimArm()
        a.speedup = 8
        await a.connect()
        route = [[2.0, 0.7, 0.0], [0.6, 0.0, 0.3], [1.5, 0.6, 0.3]]
        await a.run_route(route)
        out = list(a.pose)
        assert math.hypot(out[0], out[1]) > 0.2
        await a.run_route(route, reverse=True)
        x, y, h = a.pose
        assert math.hypot(x, y) < 0.05 and abs(h) < 0.05
    run(go())


PIN = {"X-Reach-Pin": "2468"}


def wait_for(h, pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        s = h.get("/api/state").json()
        if pred(s):
            return s
        time.sleep(0.1)
    raise AssertionError(f"timed out; last state: {s['job']}")


def test_teach_by_driving_then_fetch_by_voice(server):  # noqa: F811
    h = httpx.Client(base_url=server, timeout=10)
    assert h.post("/api/teach", json={"do": "start", "item": "inhaler"}, headers=PIN).json()["teaching"] == "inhaler"
    for _ in range(10):  # 1 s of joystick at 10 Hz
        h.post("/api/drive", json={"thr": 0.8, "turn": 0.2}, headers=PIN)
        time.sleep(0.1)
    h.post("/api/drive", json={"thr": 0, "turn": 0}, headers=PIN)
    h.post("/api/joint", json={"pose": [-20, 50, 40, 70, 0, 8]}, headers=PIN)
    saved = h.post("/api/teach", json={"do": "save", "medication": True}, headers=PIN).json()
    assert saved["saved"] == "inhaler" and saved["segments"] >= 1
    s = wait_for(h, lambda s: s["job"] and s["job"]["stage"] == "done")  # it drove itself back home
    assert abs(s["arm"]["base_pose"][1]) < 0.03

    r = h.post("/api/alexa", json={"text": "Can you bring me my inhaler please"}, headers=PIN).json()
    assert r["calls"][0]["tool"] == "fetch_item" and "On my way" in r["reply"], r
    s = wait_for(h, lambda s: s["job"] and s["job"]["stage"] == "holding it out")
    assert abs(s["arm"]["base_pose"][1]) < 0.03  # retraced the route: back by the person
    r = h.post("/api/alexa", json={"text": "thanks, I've got it"}, headers=PIN).json()
    assert r["calls"][0]["tool"] == "got_it"
    wait_for(h, lambda s: s["job"]["stage"] == "done")
    r = h.post("/api/alexa", json={"text": "did I take my meds today?"}, headers=PIN).json()
    assert "inhaler" in r["reply"]
    tools = [c["tool"] for c in h.get("/api/state").json()["calls"]]
    assert {"fetch_item", "got_it", "medication_today"} <= set(tools)


def test_alexa_sim_stop_wins(server):  # noqa: F811
    h = httpx.Client(base_url=server, timeout=10)
    h.post("/api/alexa", json={"text": "get me the water"}, headers=PIN)
    r = h.post("/api/alexa", json={"text": "wait, stop!"}, headers=PIN).json()
    assert r["calls"][0]["tool"] == "stop_arm" and r["reply"] == "Stopped."
