"""Robot backends. Every robot takes the same six-number arm pose, in its own
calibrated degrees:

    [base_pan, shoulder, elbow, wrist_flex, wrist_roll, gripper]

and, if it has wheels, a drive command (throttle, turn) in -1..1 with turn > 0
= left, the Thenar Walker's convention. Arm motion is streamed as small
interpolated steps (25 Hz) capped per joint, so nothing jumps.

Backends:
  SimArm     no hardware: arm pose plus a simulated skid-steer base, for the 3-D view
  WalkerArm  the Thenar Walker (RoboReach robot) over its Wi-Fi WebSocket  -> walker.py
  SO101Arm   a bench SO-101 follower over USB serial ("Q a b c d e f", "ARM", "STOP")
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

import numpy as np

log = logging.getLogger("reach.arm")

LO = np.array([-85, -80, -80, -80, -85, 0], float)
HI = np.array([85, 80, 80, 80, 85, 70], float)
HOME = np.array([0, -25, 35, 0, 0, 20], float)
VMAX = np.array([60, 40, 50, 80, 90, 120], float)  # deg/s per joint: slow and gentle near people
RATE = 25.0
JOINTS = ["base", "shoulder", "elbow", "wrist_flex", "wrist_roll", "gripper"]

# Thenar Walker drive mixing (firmware/libraries/ThenarLink/src/tw_robot.h), used by the simulator.
SPEED_SCALE, TURN_SCALE, SPIN_MIN, DEADBAND = (0.35, 0.65, 1.0), (0.5, 0.75, 1.0), 0.35, 0.05
TOP_SPEED_MS = 0.34  # wheel surface speed at full PWM
TRACK_M = 0.17


def mix(thr: float, turn: float, speed: int = 1) -> tuple[float, float]:
    """Left/right wheel commands -1..1 from throttle/turn, exactly as the robot firmware mixes them."""
    t, r = max(-1, min(1, thr)), max(-1, min(1, turn))
    t = 0 if abs(t) < DEADBAND else t
    r = 0 if abs(r) < DEADBAND else r
    if t == 0 and r != 0:
        mag = SPIN_MIN + (TURN_SCALE[speed] - SPIN_MIN) * abs(r)
        left = -mag if r > 0 else mag
        return left, -left
    left, right = t * SPEED_SCALE[speed] - r * TURN_SCALE[speed], t * SPEED_SCALE[speed] + r * TURN_SCALE[speed]
    m = max(1.0, abs(left), abs(right))
    return left / m, right / m


class Stopped(Exception):
    """The robot was stopped (by voice, the web page, or a fault)."""


class Arm:
    name = "arm"
    speedup = 1.0  # tests run the simulator faster than real time
    has_base = False

    def __init__(self):
        self.lo, self.hi = LO.copy(), HI.copy()
        self.home = HOME.copy()
        self.q = HOME.copy()
        self.target = HOME.copy()
        self.stop_flag = asyncio.Event()
        self.drive_cmd = (0.0, 0.0)

    @property
    def grip_open(self) -> float:
        return float(self.hi[5])

    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.lo, self.hi)

    async def connect(self) -> None:
        pass

    async def _write(self, q: np.ndarray) -> None:
        raise NotImplementedError

    async def _drive(self, thr: float, turn: float) -> None:
        raise Stopped("this robot has no wheels")

    def _check(self) -> None:
        if self.stop_flag.is_set():
            raise Stopped()

    async def move(self, target, speed: float = 1.0) -> None:
        """Interpolate to target at <= VMAX*speed per joint. Raises Stopped."""
        target = self.clamp(target)
        self.target = target
        dt = 1 / RATE
        while True:
            self._check()
            d = target - self.q
            lim = VMAX * speed * self.speedup * dt
            self.q = self.q + np.clip(d, -lim, lim)
            await self._write(self.q)
            if np.all(np.abs(d) < 0.05):
                return
            await asyncio.sleep(dt)

    async def gripper(self, opening: float) -> None:
        q = self.q.copy()
        q[5] = opening
        await self.move(q)

    async def drive(self, thr: float, turn: float, seconds: float) -> None:
        """Hold a drive command for `seconds` (open loop), checking for stop every 20 ms."""
        end = time.monotonic() + seconds / self.speedup
        try:
            while time.monotonic() < end:
                self._check()
                await self._drive(thr, turn)
                await asyncio.sleep(0.02)
        finally:
            await self._drive(0.0, 0.0)

    async def run_route(self, route: list[list[float]], reverse: bool = False) -> None:
        """Replay a taught route [[seconds, thr, turn], ...]. Reverse retraces it backwards:
        same segments in reverse order with throttle and turn negated (the turn rate of a
        skid-steer base depends only on the turn term, so negating it undoes the heading change)."""
        segs = [[s, -t, -r] for s, t, r in reversed(route)] if reverse else route
        for s, t, r in segs:
            await self.drive(t, r, s)

    async def stop(self) -> None:
        self.stop_flag.set()

    def clear_stop(self) -> None:
        self.stop_flag.clear()

    def state(self) -> dict:
        return {"backend": self.name, "joints": dict(zip(JOINTS, np.round(self.q, 1).tolist())),
                "lo": self.lo.round(1).tolist(), "hi": self.hi.round(1).tolist(), "has_base": self.has_base}


class SimArm(Arm):
    """Arm plus a simulated Thenar Walker base, so routes can be taught and replayed without hardware.

    The base moves along exact arcs (constant speed and turn rate per command), advanced by the
    commanded segment durations, so replaying a route and its reverse lands back where it began;
    a real base on a real floor will drift, which is why routes are kept short."""

    name = "sim"
    has_base = True

    def __init__(self):
        super().__init__()
        self.pose = [0.0, 0.0, 0.0]  # x (right), y (forward), heading (rad, + = left)
        self._t = time.monotonic()

    def _advance(self, thr: float, turn: float, dt: float, pose: list[float] | None = None) -> list[float]:
        left, right = mix(thr, turn)
        v = (left + right) / 2 * TOP_SPEED_MS
        w = (right - left) * TOP_SPEED_MS / TRACK_M
        x, y, h = pose or self.pose
        if abs(w) < 1e-9:
            return [x - v * math.sin(h) * dt, y + v * math.cos(h) * dt, h]
        h2 = h + w * dt
        return [x + v / w * (math.cos(h2) - math.cos(h)), y + v / w * (math.sin(h2) - math.sin(h)), h2]

    def _commit(self) -> None:
        now = time.monotonic()
        self.pose = self._advance(*self.drive_cmd, (now - self._t) * self.speedup)
        self._t = now

    async def _write(self, q):
        return

    async def _drive(self, thr, turn):
        self._commit()
        self.drive_cmd = (thr, turn)

    async def drive(self, thr: float, turn: float, seconds: float) -> None:
        self._commit()
        self.drive_cmd = (thr, turn)
        end = self._t + seconds / self.speedup
        try:
            while (now := time.monotonic()) < end:
                self._check()
                await asyncio.sleep(min(0.02, end - now))
        except Stopped:
            self._commit()
            self.drive_cmd = (0.0, 0.0)
            raise
        self.pose = self._advance(thr, turn, seconds)  # exactly the commanded segment
        self._t = time.monotonic()
        self.drive_cmd = (0.0, 0.0)

    def state(self) -> dict:
        s = super().state()
        live = self._advance(*self.drive_cmd, (time.monotonic() - self._t) * self.speedup)
        s["base_pose"] = [round(live[0], 3), round(live[1], 3), round(live[2], 3)]
        s["drive"] = list(self.drive_cmd)
        return s


class SO101Arm(Arm):
    """A bench SO-101 follower on USB. Refuses to arm unless it acknowledges STOP first,
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
