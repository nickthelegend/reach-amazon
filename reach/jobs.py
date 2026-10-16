"""Fetch jobs. Alexa+ wants every tool call answered in well under a second, but
fetching takes a while, so a fetch starts here and runs in the background;
the tools report on it.

A fetch: open gripper -> approach above the item -> descend -> close ->
lift -> hold it out to the person -> wait for "I've got it" (or a timeout) ->
open -> return home.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .arm import HOME, Arm, Stopped
from .world import HANDOVER, Place, World

log = logging.getLogger("reach.jobs")

GRIP_CLOSED = 8.0
HOLD_TIMEOUT_S = 90.0


@dataclass
class Job:
    item: str
    medication: bool
    stage: str = "starting"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    outcome: str | None = None  # taken | not_taken | stopped | failed
    detail: str = ""

    def view(self) -> dict:
        return {"item": self.item, "stage": self.stage, "outcome": self.outcome, "detail": self.detail,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


def approach_of(pose: list[float]) -> np.ndarray:
    """Hover above the grasp: shoulder back, elbow up, gripper open."""
    a = np.array(pose, float)
    a[1] -= 18
    a[2] += 12
    a[5] = max(a[5], 55)
    return a


class Runner:
    def __init__(self, arm: Arm, world: World):
        self.arm, self.world = arm, world
        self.job: Job | None = None
        self.task: asyncio.Task | None = None
        self._taken = asyncio.Event()

    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def start_fetch(self, place: Place) -> Job:
        if self.busy():
            raise RuntimeError(f"already fetching the {self.job.item}")
        self.arm.clear_stop()
        self._taken.clear()
        self.job = Job(place.name, place.medication)
        self.task = asyncio.create_task(self._fetch(place, self.job))
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
        try:
            await self._stage(job, f"reaching for the {place.name}")
            await self.arm.move(approach_of(place.pose))
            grasp = np.array(place.pose, float)
            grasp[5] = max(grasp[5], 55)
            await self.arm.move(grasp, speed=0.5)
            await self._stage(job, "picking it up")
            await self.arm.gripper(GRIP_CLOSED)
            lifted = approach_of(place.pose)
            lifted[5] = GRIP_CLOSED
            await self.arm.move(lifted, speed=0.6)
            await self._stage(job, "bringing it to you")
            hand = np.array(HANDOVER, float)
            hand[5] = GRIP_CLOSED
            await self.arm.move(hand, speed=0.7)
            await self._stage(job, "holding it out")
            # Wait for "I've got it", but a stop must still win while the arm is standing still.
            taken = asyncio.create_task(self._taken.wait())
            stopped = asyncio.create_task(self.arm.stop_flag.wait())
            done, pending = await asyncio.wait({taken, stopped}, timeout=HOLD_TIMEOUT_S,
                                               return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if stopped in done:
                raise Stopped("stopped on request while holding it out")
            if taken in done:
                await self.arm.gripper(60)
                job.outcome = "taken"
            else:
                job.outcome = "not_taken"
                job.detail = "nobody took it, so it went back"
                await self.arm.move(lifted)
                down = np.array(place.pose, float)
                down[5] = GRIP_CLOSED
                await self.arm.move(down, speed=0.5)
                await self.arm.gripper(60)
            await self._stage(job, "going back")
            await self.arm.move(HOME)
            await self._stage(job, "done")
        except Stopped as e:
            job.outcome, job.stage, job.detail = "stopped", "stopped", str(e) or "stopped on request"
        except Exception as e:  # noqa: BLE001 - a fault must end the job, not the server
            log.exception("fetch failed")
            job.outcome, job.stage, job.detail = "failed", "failed", str(e)
            await self.arm.stop()
        finally:
            job.finished = time.time()
            self.world.record(job.item, job.outcome or "failed", job.medication)
