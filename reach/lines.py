"""Everything Reach says that can be known ahead of time.

The tools in server.py build their replies with these functions, and voice_lines() lists
every reply they can produce for the items Reach knows, so the voice can render them in
advance (tts.py) without the two ever drifting apart. No imports from the rest of reach:
tests and jobs.py can use it without starting a server.
"""

from __future__ import annotations

STOPPED = "Stopped."
LETTING_GO = "Okay, letting go."
NOT_HOLDING = "I'm not holding anything right now."
RESTING = "I'm resting and ready."
NO_MEDS = "I haven't handed you any medication today."
CANT_LEARN = "I can't learn a spot while I'm moving."
DIDNT_CATCH = "Sorry, I didn't catch that. You can say, bring me my pills."


def items_sentence(names) -> str:
    names = sorted(names)
    return ", ".join(names[:-1]) + (" and " + names[-1] if len(names) > 1 else (names[0] if names else "nothing yet"))


def can_bring(names) -> str:
    return f"I can bring your {items_sentence(names)}."


def fetch_reply(name: str, route, has_base: bool) -> str:
    if route and has_base:
        secs = sum(s for s, _, _ in route) * 2 + 20
        return f"On my way to get your {name}. I'll bring it over in about {round(secs / 10) * 10} seconds. Tell me when you've got it."
    return f"On my way to get your {name}. I'll hold it out in about fifteen seconds. Tell me when you've got it."


def busy_reply(item: str) -> str:
    return f"I'm still busy with the {item}. Say stop if you want me to stop."


def got_it_busy(stage: str) -> str:
    return f"I'm still {stage}. I'll tell you when it's ready to take."


def status_busy(stage: str, item: str) -> str:
    return f"I'm {stage}." if stage != "holding it out" else f"I'm holding out your {item}. Say you've got it when you take it."


def status_done(outcome: str | None, item: str, detail: str = "") -> str:
    return {"taken": f"I brought your {item}.", "not_taken": f"Nobody took the {item}, so I put it back.",
            "stopped": "I stopped.", "failed": f"Something went wrong getting the {item}: {detail}."}.get(outcome or "", "I'm ready.")


def holding_line(item: str) -> str:
    """Said when the arm holds the item out (the simulator page speaks it; real Alexa+ would notify)."""
    return f"Here's your {item}. Take it and say \"I've got it\"."


def remember_reply(name: str) -> str:
    return f"Got it. I'll remember where your {name} {'are' if name.endswith('s') else 'is'}."


def fetch_stages(name: str, drive: bool) -> list[str]:
    """Every stage a fetch passes through while busy (jobs.Runner._fetch)."""
    return ["starting", *([f"going to the {name}"] if drive else []), f"reaching for the {name}", "picking it up",
            "bringing it to you", "holding it out", "going back", "done"]


RETURN_STAGES = ["learned it, going back home", "done"]  # jobs.Runner.start_return


def voice_lines(places, has_base: bool) -> list[str]:
    """The catalog, most important first: safety and handover, then fetches, then status, then the rest.
    places: objects with .name and .route (world.Place)."""
    places = sorted(places, key=lambda p: p.name)
    names = [p.name for p in places]
    out = [STOPPED, LETTING_GO, *(holding_line(n) for n in names)]
    out += [x for p in places for x in (fetch_reply(p.name, p.route, has_base), busy_reply(p.name))]
    for p in places:
        stages = fetch_stages(p.name, bool(p.route) and has_base) + RETURN_STAGES
        for s in dict.fromkeys(stages):
            out += [status_busy(s, p.name), got_it_busy(s)]
        out += [status_done(o, p.name) for o in ("taken", "not_taken")]
    out += [RESTING, NOT_HOLDING, status_done("stopped", ""), status_done(None, ""), NO_MEDS, CANT_LEARN,
            can_bring(names), DIDNT_CATCH, *(remember_reply(n) for n in names)]
    return list(dict.fromkeys(out))
