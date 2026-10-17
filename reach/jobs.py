"""Fetch jobs. Alexa+ wants every tool call answered in well under a second, but a
fetch takes a while, so it starts here and runs in the background; the tools
report on it.

A fetch:
  1. drive the taught route from home to the item (if the robot has wheels and a route)
  2. hover above the grip with the gripper open, descend, close, lift
  3. retrace the route back home
  4. hold the item out (taught handover pose) until the person says they have it
  5. let go and fold back to home

"Hover" is the grip pose moved 35% of the way toward the home pose: home is the
folded, upright pose, so this lifts the gripper clear on any robot without
knowing each joint's sign convention.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .arm import Arm, Stopped
from .lines import holding_line
from .world import HANDOVER, Place, World

log = logging.getLogger("reach.jobs")

HOVER = 0.35


@dataclass
class Job:
    item: str
    medication: bool
    stage: str = "starting"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    outcome: str | None = None  # taken | stopped | failed
    detail: str = ""

    def view(self) -> dict:
        return {"item": self.item, "stage": self.stage, "outcome": self.outcome, "detail": self.detail,
                "seconds": round((self.finished or time.time()) - self.started, 1),
                "say": holding_line(self.item) if self.stage == "holding it out" else None}


def hover_of(grip: np.ndarray, home: np.ndarray, gripper: float) -> np.ndarray:
    h = grip.copy()
    h[1:4] = grip[1:4] + HOVER * (home[1:4] - grip[1:4])
    h[5] = gripper
    return h


class Runner:
    def __init__(self, arm: Arm, world: World):
        self.arm, self.world = arm, world
        self.job: Job | None = None
        self.task: asyncio.Task | None = None
        self._taken = asyncio.Event()

    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def home_pose(self) -> np.ndarray:
        return self.arm.clamp(self.world.home if self.world.home else self.arm.home)

    def handover_pose(self) -> np.ndarray:
        if self.world.handover:
            return self.arm.clamp(self.world.handover)
        return self.arm.clamp(HANDOVER) if self.arm.name == "sim" else self.home_pose()

    def start_fetch(self, place: Place) -> Job:
        if self.busy():
            raise RuntimeError(f"already fetching the {self.job.item}")
        self.arm.clear_stop()
        self._taken.clear()
        self.job = Job(place.name, place.medication)
        self.task = asyncio.create_task(self._fetch(place, self.job))
        return self.job

    def start_return(self, route: list[list[float]], item: str) -> Job:
        """After teaching: drive back home the way it came."""
        self.arm.clear_stop()
        self.job = Job(item, False, stage="learned it, going back home")

        async def back(job: Job):
            try:
                await self.arm.move(self.home_pose())
                await self.arm.run_route(route, reverse=True)
                job.outcome, job.stage = "taught", "done"
            except Stopped as e:
                job.outcome, job.stage, job.detail = "stopped", "stopped", str(e)
            finally:
                job.finished = time.time()

        self.task = asyncio.create_task(back(self.job))
        return self.job

    def taken(self) -> bool:
        """The person has the item: open the gripper. True if something was being held out."""
        if self.job and self.job.stage == "holding it out":
            self._taken.set()
            return True
        return False

    async def stop(self) -> None:
        await self.arm.stop()

    async def _stage(self, job: Job, stage: str) -> None:
        job.stage = stage
        log.info("%s: %s", job.item, stage)

    async def _fetch(self, place: Place, job: Job) -> None:
        arm = self.arm
        drive = arm.has_base and bool(place.route)
        try:
            home = self.home_pose()
            grip = arm.clamp(place.pose)
            closed = float(grip[5])
            if drive:
                await self._stage(job, f"going to the {place.name}")
                await arm.move(home)
                await arm.run_route(place.route)
            await self._stage(job, f"reaching for the {place.name}")
            await arm.move(hover_of(grip, home, arm.grip_open))
            g = grip.copy()
            g[5] = arm.grip_open
            await arm.move(g, speed=0.5)
            await self._stage(job, "picking it up")
            await arm.gripper(closed)
            await arm.move(hover_of(grip, home, closed), speed=0.6)
            if drive:
                await self._stage(job, "bringing it to you")
                tuck = home.copy()
                tuck[5] = closed
                await arm.move(tuck, speed=0.6)  # carry it folded in while driving
                await arm.run_route(place.route, reverse=True)
            else:
                await self._stage(job, "bringing it to you")
            hand = self.handover_pose().copy()
            hand[5] = closed
            await arm.move(hand, speed=0.7)
            await self._stage(job, "holding it out")
            # Wait for "I've got it"; a stop must still win while the arm is standing still.
            taken = asyncio.create_task(self._taken.wait())
            stopped = asyncio.create_task(arm.stop_flag.wait())
            done, pending = await asyncio.wait({taken, stopped}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if stopped in done:
                raise Stopped("stopped on request while holding it out")
            await arm.gripper(arm.grip_open)
            job.outcome = "taken"
            await self._stage(job, "going back")
            await arm.move(home)
            await self._stage(job, "done")
        except Stopped as e:
            job.outcome, job.stage, job.detail = "stopped", "stopped", str(e) or "stopped on request"
        except Exception as e:  # noqa: BLE001 - a fault must end the job, not the server
            log.exception("fetch failed")
            job.outcome, job.stage, job.detail = "failed", "failed", str(e)
            await arm.stop()
        finally:
            job.finished = time.time()
            self.world.record(job.item, job.outcome or "failed", job.medication)
