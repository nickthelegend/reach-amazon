"""A stand-in Thenar Walker that speaks the robot firmware's WebSocket protocol, for tests and
demos without the robot. Same semantics as firmware/thenar_phone: config on connect, telemetry
every 100 ms, 300 ms silence -> wheels stop, STOP latches the wheels until the stick is centred,
arm targets slewed at 120 deg/s and clamped to the calibrated ends.

    python -m reach.mock_walker --port 8181
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

LO = [-90.0, -10.0, -95.0, -95.0, -90.0, -5.0]
HI = [90.0, 95.0, 95.0, 95.0, 90.0, 60.0]


class MockWalker:
    def __init__(self):
        self.goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cur = list(self.goal)
        self.on = 0
        self.speed = 1
        self.thr = self.trn = 0.0
        self.latch = False
        self.last = time.monotonic()
        self.log: list[str] = []
        self.silent_stops = 0

    def cfg(self) -> str:
        return "C" + json.dumps({"ssid": "Thenar_TF", "cal": 1, "pca": 1, "usable": 63, "on": self.on, "sp": self.speed,
                                 "lo": LO, "hi": HI, "g": self.goal})

    def handle(self, c: str) -> str | None:
        self.last = time.monotonic()
        self.log.append(c)
        p = c.split()
        k = p[0]
        if k == "D":
            self.thr, self.trn = int(p[1]) / 100, int(p[2]) / 100
            if self.latch and abs(self.thr) < 0.05 and abs(self.trn) < 0.05:
                self.latch = False
        elif k == "S":
            self.speed = max(0, min(2, int(p[1])))
        elif k == "X":
            self.latch = True
            self.goal = list(self.cur)
        elif k == "M":
            self.on = int(p[1]) & 63
        elif k == "A":
            self.on = 63 if p[1] == "1" else 0
        elif k == "Q":
            i, v = int(p[1]), float(p[2])
            self.goal[i] = max(LO[i], min(HI[i], v))
        elif k == "G":
            return self.cfg()
        elif k == "P":
            return c
        return None

    async def tick(self, ws) -> None:
        while True:
            await asyncio.sleep(0.1)
            if time.monotonic() - self.last > 0.3 and (self.thr or self.trn):
                self.thr = self.trn = 0.0  # phone silent -> wheels stop
                self.silent_stops += 1
            for i in range(6):
                if self.on >> i & 1:
                    d = self.goal[i] - self.cur[i]
                    self.cur[i] += max(-12.0, min(12.0, d))  # 120 deg/s at 10 Hz
            await ws.send(f"T 11.80 {self.on} {self.speed} {int(self.latch)} " + " ".join(f"{v:.1f}" for v in self.cur))

    @property
    def moving(self) -> bool:
        return not self.latch and (abs(self.thr) >= 0.05 or abs(self.trn) >= 0.05)


async def serve(port: int, robot: MockWalker | None = None):
    import websockets

    robot = robot or MockWalker()

    async def conn(ws):
        await ws.send(robot.cfg())
        t = asyncio.create_task(robot.tick(ws))
        try:
            async for msg in ws:
                r = robot.handle(msg)
                if r:
                    await ws.send(r)
        finally:
            t.cancel()

    return robot, await websockets.serve(conn, "127.0.0.1", port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8181)
    a = ap.parse_args()

    async def main():
        await serve(a.port)
        print(f"mock Thenar Walker on ws://127.0.0.1:{a.port}/")
        await asyncio.Future()

    asyncio.run(main())
