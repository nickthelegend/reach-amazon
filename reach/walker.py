"""The Thenar Walker (RoboReach robot) as a REACH body, over its own Wi-Fi.

Protocol: firmware/thenar_phone/thenar_phone.ino in nickthelegend/thenar-walker. The robot is a
Wi-Fi access point (http://192.168.4.1) with ONE WebSocket on port 81 (newest client wins):

  to robot:   D thr turn (-100..100, turn > 0 = left)   S speed 0..2   X (STOP)   A 1|0 (arm on/limp)
              M mask   Q joint deg (target)   V joint deg_per_s   P n (ping)   G (config)
  from robot: C{json config: lo, hi, g, usable, on, sp, cal, pca}   T bat on speed latch q0..q5 (every 100 ms)

The firmware stops the wheels if it hears nothing for 300 ms, so this sends the current drive command
every 40 ms whether or not it changed: if REACH dies, the robot stops by itself. Joint angles are the
robot's calibrated degrees, clamped to its calibrated safe ends (reported in C).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np

from .arm import Arm, Stopped

log = logging.getLogger("reach.walker")

HEARTBEAT_S = 0.04


class WalkerArm(Arm):
    name = "walker"
    has_base = True

    def __init__(self, url: str = "ws://192.168.4.1:81/", speed: int = 0):
        super().__init__()
        self.url = url
        self.speed = speed  # 0 = slow: indoors, near a person
        self.ws = None
        self.cfg: dict = {}
        self.tele: dict = {}
        self._tasks: list[asyncio.Task] = []
        self._ready = asyncio.Event()
        self._sent_q = [None] * 6

    async def connect(self) -> None:
        import websockets

        self.ws = await websockets.connect(self.url, open_timeout=5, ping_interval=None, max_size=4096)
        self._tasks = [asyncio.create_task(self._rx()), asyncio.create_task(self._heartbeat())]
        await self._send("G")
        await asyncio.wait_for(self._ready.wait(), 5)
        c = self.cfg
        if not c.get("cal") or not c.get("pca"):
            raise RuntimeError(f"robot not ready: calibration={c.get('cal')} servo driver={c.get('pca')}")
        self.lo, self.hi = np.array(c["lo"], float), np.array(c["hi"], float)
        self.q = self.target = np.array(c["g"], float)
        self.home = self.q.copy()
        await self._send(f"S {self.speed}")
        # Switch joints on one at a time (each jumps to its target): the base and wrist first, the
        # shoulder (the heavy one) last, 300 ms apart, so the servo rail never sees four starting at once.
        usable, mask = int(c["usable"]), 0
        for i in (0, 4, 3, 5, 2, 1):
            if usable >> i & 1:
                mask |= 1 << i
                await self._send(f"M {mask}")
                await asyncio.sleep(0.3)
        log.info("Thenar Walker ready: joints on %s, battery %.2f V", bin(mask), self.tele.get("bat", 0))

    async def _send(self, s: str) -> None:
        if not self.ws:
            raise Stopped("robot not connected")
        try:
            await self.ws.send(s)
        except Exception as e:  # noqa: BLE001 - a dropped link is a stop
            self.stop_flag.set()
            raise Stopped(f"robot link lost: {e}") from e

    async def _rx(self) -> None:
        try:
            async for msg in self.ws:
                if msg.startswith("C"):
                    self.cfg = json.loads(msg[1:])
                    self._ready.set()
                elif msg.startswith("T "):
                    p = msg.split()
                    self.tele = {"bat": float(p[1]), "on": int(p[2]), "sp": int(p[3]), "latch": int(p[4]),
                                 "q": [float(x) for x in p[5:11]]}
        finally:
            log.warning("robot link closed")
            self.stop_flag.set()

    async def _heartbeat(self) -> None:
        while True:
            thr, turn = self.drive_cmd
            await self._send(f"D {round(thr * 100)} {round(turn * 100)}")
            await asyncio.sleep(HEARTBEAT_S)

    async def _write(self, q: np.ndarray) -> None:
        for i in range(6):
            v = round(float(q[i]), 1)
            if self._sent_q[i] != v:
                await self._send(f"Q {i} {v}")
                self._sent_q[i] = v

    async def _drive(self, thr: float, turn: float) -> None:
        changed = (thr, turn) != self.drive_cmd
        self.drive_cmd = (thr, turn)  # the heartbeat keeps repeating it
        if changed:  # but a change (above all a stop) goes out now, not on the next beat
            await self._send(f"D {round(thr * 100)} {round(turn * 100)}")

    async def stop(self) -> None:
        await super().stop()
        self.drive_cmd = (0.0, 0.0)
        try:
            await self._send("X")
        except Stopped:
            pass

    def state(self) -> dict:
        s = super().state()
        s["battery_v"] = self.tele.get("bat")
        s["robot_q"] = self.tele.get("q")
        s["drive"] = list(self.drive_cmd)
        return s
