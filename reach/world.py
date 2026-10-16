"""What the arm knows about the room: where things are, and what it has handed over.

Places are taught, not hard-coded: move the arm (or the leader arm) to an
item and say "remember this is my pills". Each place stores the grasp pose;
the approach pose is derived by lifting the shoulder and elbow above it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PLACES = {
    # A starter layout for the demo table, in joint degrees (see arm.py).
    "pills": {"pose": [-40, 25, -10, 45, 0, 60], "medication": True, "aliases": ["medicine", "pill organizer", "tablets", "meds"]},
    "water": {"pose": [-15, 30, -20, 40, 0, 65], "medication": False, "aliases": ["water bottle", "bottle", "drink"]},
    "glasses": {"pose": [20, 28, -15, 50, 90, 55], "medication": False, "aliases": ["spectacles", "reading glasses"]},
    "remote": {"pose": [45, 32, -25, 45, 90, 50], "medication": False, "aliases": ["tv remote", "remote control"]},
    "phone": {"pose": [65, 26, -12, 48, 0, 55], "medication": False, "aliases": ["mobile", "cell phone"]},
}
HANDOVER = [0, -5, 55, -30, 0, 30]  # held out toward the person, gripper still closed on the item


@dataclass
class Place:
    name: str
    pose: list[float]
    medication: bool = False
    aliases: list[str] = field(default_factory=list)


class World:
    def __init__(self, path: Path):
        self.path = path
        self.places: dict[str, Place] = {}
        self.log: list[dict] = []  # deliveries
        self.load()

    def load(self) -> None:
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.places = {k: Place(k, **v) for k, v in d.get("places", {}).items()}
            self.log = d.get("log", [])
        else:
            self.places = {k: Place(k, **v) for k, v in DEFAULT_PLACES.items()}
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        d = {"places": {k: {kk: vv for kk, vv in asdict(p).items() if kk != "name"} for k, p in self.places.items()},
             "log": self.log[-500:]}
        self.path.write_text(json.dumps(d, indent=1))

    def find(self, spoken: str) -> Place | None:
        s = spoken.lower().strip().removeprefix("my ").removeprefix("the ")
        for p in self.places.values():
            if s == p.name or s in p.aliases:
                return p
        for p in self.places.values():  # "my blood pressure pills" -> pills
            if p.name in s or any(a in s for a in p.aliases):
                return p
        return None

    def remember(self, name: str, pose: list[float], medication: bool = False) -> Place:
        key = name.lower().strip().removeprefix("my ").removeprefix("the ")
        p = self.places.get(key) or Place(key, pose)
        p.pose = [round(x, 1) for x in pose]
        p.medication = medication or p.medication
        self.places[key] = p
        self.save()
        return p

    def record(self, item: str, outcome: str, medication: bool) -> None:
        self.log.append({"t": time.time(), "item": item, "outcome": outcome, "medication": medication})
        self.save()

    def medication_today(self) -> list[dict]:
        start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        return [e for e in self.log if e["medication"] and e["outcome"] == "taken" and e["t"] >= start]
