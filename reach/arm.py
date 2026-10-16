"""Arm backends. Every arm takes the same six-number pose, in degrees:

    [base, shoulder, elbow, wrist_flex, wrist_roll, gripper]

with the SO-101 limits below (gripper 0 = closed, 70 = open). Motion is
streamed as small interpolated steps at 50 Hz, capped per joint, so a fetch
never jumps.

Backends:
  SimArm     no hardware; the pose is published for the 3-D view
  SO101Arm   the Thenar SO-101 / MG996R follower over USB serial
             (firmware protocol: "Q a b c d e f", "ARM", "STOP")
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

log = logging.getLogger("reach.arm")

LO = np.array([-85, -80, -80, -80, -85, 0], float)
HI = np.array([85, 80, 80, 80, 85, 70], float)
HOME = np.array([0, -25, 35, 0, 0, 20], float)
VMAX = np.array([60, 40, 50, 80, 90, 120], float)  # deg/s per joint: slow and gentle near people
RATE = 50.0
JOINTS = ["base", "shoulder", "elbow", "wrist_flex", "wrist_roll", "gripper"]


class Stopped(Exception):
    """The arm was stopped (by voice, the web page, or a fault)."""


class Arm:
    name = "arm"
    speedup = 1.0  # tests run the simulator faster than real time

    def __init__(self):
        self.q = HOME.copy()
        self.target = HOME.copy()
        self.stop_flag = asyncio.Event()
        self.listeners: list = []

    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, float), LO, HI)

    async def connect(self) -> None:
        pass

    async def _write(self, q: np.ndarray) -> None:
        raise NotImplementedError

    async def move(self, target, speed: float = 1.0) -> None:
        """Interpolate to target at <= VMAX*speed per joint. Raises Stopped."""
        target = self.clamp(target)
        self.target = target
        dt = 1 / RATE
        while True:
            if self.stop_flag.is_set():
                raise Stopped()
            d = target - self.q
            lim = VMAX * speed * self.speedup * dt
            step = np.clip(d, -lim, lim)
            self.q = self.q + step
            await self._write(self.q)
            for cb in self.listeners:
                cb(self.q)
            if np.all(np.abs(d) < 0.05):
                return
            await asyncio.sleep(dt)

    async def gripper(self, opening: float) -> None:
        q = self.q.copy()
        q[5] = opening
        await self.move(q, speed=1.0)

    async def stop(self) -> None:
        self.stop_flag.set()

    def clear_stop(self) -> None:
        self.stop_flag.clear()

    def state(self) -> dict:
        return {"backend": self.name, "joints": dict(zip(JOINTS, np.round(self.q, 1).tolist()))}


class SimArm(Arm):
    name = "sim"

    async def _write(self, q):
        return


class SO101Arm(Arm):
    """The Thenar follower. Refuses to arm unless it acknowledges STOP first,
    and every streamed pose passes the table guard from thenar-arms."""

    name = "so101"

    def __init__(self, port: str, guard=None):
        super().__init__()
        self.port = port
        self.guard = guard
        self.ser = None

    async def connect(self) -> None:
        import serial

        self.ser = serial.Serial(self.port, 115200, timeout=0.02, write_timeout=0.1)
        await asyncio.sleep(2)
        self.ser.reset_input_buffer()
        self.ser.write(b"STOP\n")
        if not await self._expect("STOP", 2.0):
            raise RuntimeError("follower did not acknowledge STOP; refusing to arm")
        # Arm from the home pose the firmware expects (support the arm while doing this).
        await self._write(HOME)
        self.ser.write(b"ARM\n")
        if not await self._expect("ARMED", 3.0):
            raise RuntimeError("follower did not arm")
        self.q = HOME.copy()
        log.info("SO-101 armed on %s", self.port)

    async def _expect(self, word: str, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            line = self.ser.readline(256).decode(errors="replace").strip()
            if line.startswith("FAULT"):
                raise RuntimeError(line)
            if line.startswith(word):
                return True
            await asyncio.sleep(0.01)
        return False

    async def _write(self, q):
        if self.guard and not self.guard.safe(q):
            await self.stop()
            raise Stopped("pose would enter the table")
        self.ser.write(("Q " + " ".join(f"{v:.3f}" for v in q) + "\n").encode())
        reply = self.ser.readline(256).decode(errors="replace").strip()
        if reply.startswith(("FAULT", "STOP")):
            self.stop_flag.set()
            raise Stopped(reply)

    async def stop(self) -> None:
        await super().stop()
        if self.ser:
            self.ser.write(b"STOP\n")


def load_guard(thenar_arms_dir: str | None):
    """The SO-101 table guard from the thenar-arms repo, if it's on this machine."""
    if not thenar_arms_dir:
        return None
    import importlib.util
    from pathlib import Path

    p = Path(thenar_arms_dir) / "so101-mg996r/firmware/bridge.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("thenar_bridge", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TableGuard()
