"""REACH: an Alexa+ MCP server that lets someone with limited mobility ask a
robot arm to bring them things.

    "Alexa, ask Reach to bring me my pills."
    "Alexa, tell Reach I've got it."
    "Alexa, tell Reach to stop."
    "Alexa, ask Reach if I took my pills today."

Run:   python -m reach.server            (serves /mcp, OAuth, and the dashboard on :8790)
Env:   REACH_ARM=sim | walker | /dev/cu.usbserial-XXXX   which robot (default sim)
       REACH_WALKER_URL=ws://192.168.4.1:81/      the Thenar Walker's WebSocket (join its Wi-Fi first)
       REACH_BASE_URL=https://<public-url>        what Alexa+ reaches (tunnel or host)
       REACH_AUTH=0                               no OAuth (local testing with MCP Inspector)
       REACH_OWNER_PIN, REACH_CLIENT_ID, REACH_CLIENT_SECRET, REACH_REDIRECT_URIS   (auth.py)
       THENAR_ARMS_DIR                            path to thenar-arms, for the SO-101 table guard
       CHATTERBOX_URL, REACH_VOICE=0              Reach's voice (tts.py); cached under data/voice
"""

from __future__ import annotations

import asyncio
import functools
import hmac
import json
import logging
import os
import time
from pathlib import Path

import uvicorn
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from collections import deque

from . import lines
from .alexa_sim import AlexaSim
from .arm import SimArm, SO101Arm, Stopped, load_guard
from .walker import WalkerArm
from .auth import SCOPE, ReachAuth, StripWWWAuthenticate
from .jobs import Runner
from .tts import MAX_CHARS, Voice, key
from .world import World

log = logging.getLogger("reach")
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("REACH_DATA", ROOT / "data"))
PORT = int(os.getenv("PORT", "8790"))
BASE_URL = os.getenv("REACH_BASE_URL", f"http://localhost:{PORT}").rstrip("/")
AUTH_ON = os.getenv("REACH_AUTH", "1") != "0"

world = World(DATA / "world.json")
arm_spec = os.getenv("REACH_ARM", "sim")
if arm_spec == "sim":
    arm = SimArm()
elif arm_spec == "walker":
    arm = WalkerArm(os.getenv("REACH_WALKER_URL", "ws://192.168.4.1:81/"), speed=int(os.getenv("REACH_WALKER_SPEED", "0")))
else:
    arm = SO101Arm(arm_spec, load_guard(os.getenv("THENAR_ARMS_DIR")))
arm.speedup = float(os.getenv("REACH_SIM_SPEEDUP", "1")) if arm_spec == "sim" else 1.0
runner = Runner(arm, world)
voice = Voice(DATA / "voice")
_arm_ready: asyncio.Task | None = None


calls: deque = deque(maxlen=30)  # recent MCP tool calls, for the dashboard
teach: dict = {"item": None, "segments": [], "cmd": (0.0, 0.0), "since": 0.0, "last": 0.0}


async def ensure_arm() -> None:
    """Connect the robot once, on first use (the SO-101 and the Walker take a few seconds)."""
    global _arm_ready
    if _arm_ready is None or (_arm_ready.done() and _arm_ready.exception()):
        _arm_ready = asyncio.create_task(arm.connect())
        asyncio.create_task(_drive_watchdog())
    await _arm_ready


def logged(fn):
    """Record every tool call (from Alexa+, the simulator or the dashboard) with its latency."""
    @functools.wraps(fn)
    async def wrapper(*a, **k):
        t0 = time.perf_counter()
        out = await fn(*a, **k)
        calls.append({"t": time.time(), "tool": fn.__name__, "args": k, "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "result": out})
        return out
    return wrapper


auth = ReachAuth(BASE_URL, DATA / "tokens.json") if AUTH_ON else None
mcp = MCPServer(
    name="Reach",
    title="Reach: a robot arm that brings you things",
    instructions=(
        "Reach drives a small robot arm next to someone who can't easily get up. It can bring them items it "
        "knows (pills, water, glasses, the TV remote, the phone), hold them out until they take them, and keep "
        "a record of when medication was handed over. Start a fetch with fetch_item; it returns at once and the "
        "arm keeps moving. When the person says they have it, call got_it. If they say stop, wait, or sound "
        "alarmed, call stop_arm immediately. Keep replies short and spoken."
    ),
    version="0.1.0",
    auth_server_provider=auth,
    auth=AuthSettings(issuer_url=BASE_URL, resource_server_url=f"{BASE_URL}/mcp", required_scopes=[SCOPE],
                      validate_token_resource=False) if AUTH_ON else None,
)

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
ACT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def voice_lines() -> list[str]:
    """Every reply the tools can give for the items known now (see lines.py)."""
    return lines.voice_lines(world.places.values(), arm.has_base)


def _voice_catalog() -> None:
    """Tell the voice what Reach can say, so it renders those lines in the background."""
    voice.catalog(voice_lines())
    voice.start()  # (re)starts the worker if it isn't running


@mcp.tool(annotations=ACT)
@logged
async def fetch_item(item: str) -> str:
    """Bring the person an item. The arm picks it up, holds it out toward them, and waits until they say
    they have it. Returns immediately while the arm moves (about 15 seconds).

    item: what they asked for, in their words, e.g. "my pills", "water", "reading glasses", "the remote".
    """
    place = world.find(item)
    if not place:
        return f"I don't know where the {item} is. {lines.can_bring(world.places)}"
    if runner.busy():
        return lines.busy_reply(runner.job.item)
    await ensure_arm()
    runner.start_fetch(place)
    return lines.fetch_reply(place.name, place.route, arm.has_base)


@mcp.tool(annotations=ACT)
@logged
async def got_it() -> str:
    """The person has taken the item the arm is holding out. Opens the gripper and sends the arm home."""
    if runner.taken():
        return lines.LETTING_GO
    if runner.busy():
        return lines.got_it_busy(runner.job.stage)
    return lines.NOT_HOLDING


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
@logged
async def stop_arm() -> str:
    """Stop the arm right now, wherever it is. Use whenever the person says stop, wait, or sounds alarmed."""
    await runner.stop()
    return lines.STOPPED


@mcp.tool(annotations=READ)
@logged
async def arm_status() -> str:
    """What the arm is doing right now, e.g. still bringing the pills, or holding them out."""
    if not runner.job:
        return lines.RESTING
    j = runner.job
    if runner.busy():
        return lines.status_busy(j.stage, j.item)
    return lines.status_done(j.outcome, j.item, j.detail)


@mcp.tool(annotations=READ)
@logged
async def list_items() -> str:
    """The items the arm knows how to bring."""
    return lines.can_bring(world.places)


@mcp.tool(annotations=READ)
@logged
async def medication_today() -> str:
    """Whether the person has been handed their medication today, and when. For questions like
    "did I take my pills?"."""
    taken = world.medication_today()
    if not taken:
        return lines.NO_MEDS
    times = [time.strftime("%-I:%M %p", time.localtime(e["t"])) for e in taken]
    items = sorted({e["item"] for e in taken})
    return f"I handed you your {' and '.join(items)} today at {', '.join(times)}."


@mcp.tool(annotations=ACT)
@logged
async def remember_item(name: str, is_medication: bool = False) -> str:
    """Teach the arm where something is: the arm (or the leader arm) has been placed on the item by a
    helper, and this saves that spot under the given name. Only when the person or a helper asks to.

    name: what to call it, e.g. "blood pressure pills". is_medication: track it in the medication log.
    """
    if runner.busy():
        return lines.CANT_LEARN
    route = _close_segment() if teach["item"] is not None else None
    p = world.remember(name, arm.q.tolist(), is_medication, route)
    _voice_catalog()
    return lines.remember_reply(p.name)


# ---------------------------------------------------------------- dashboard ----
def _pin_ok(request: Request) -> bool:
    pin = os.getenv("REACH_OWNER_PIN", "")
    return not pin or hmac.compare_digest(request.headers.get("x-reach-pin", ""), pin)


@mcp.custom_route("/", methods=["GET"])
async def index(request: Request) -> Response:
    return FileResponse(ROOT / "web" / "index.html")


@mcp.custom_route("/api/state", methods=["GET"])
async def state(request: Request) -> Response:
    return JSONResponse({
        "arm": arm.state(),
        "job": runner.job.view() if runner.job else None,
        "busy": runner.busy(),
        "items": {k: {"medication": p.medication, "pose": p.pose, "route": p.route} for k, p in world.places.items()},
        "home": runner.home_pose().round(1).tolist(), "handover": runner.handover_pose().round(1).tolist(),
        "medication_today": world.medication_today(),
        "log": world.log[-12:],
        "calls": list(calls)[-12:],
        "teach": {"item": teach["item"], "segments": len(teach["segments"]),
                  "seconds": round(sum(x[0] for x in teach["segments"]), 1)},
        "auth": AUTH_ON,
        "alexa_engine": alexa.engine,
        "voice": voice.status(),
    })


@mcp.custom_route("/api/voice", methods=["GET"])
async def api_voice(request: Request) -> Response:
    """?text=<a line Reach says> -> 200 its Chatterbox WAV | 204 not rendered yet (moved to the front of
    the queue; use the browser's voice this time) | 404 not a line Reach knows, or the voice is off.
    Never waits on Chatterbox. No PIN: only catalog lines can be queued."""
    text = key(request.query_params.get("text", ""))
    if not text or len(text) > MAX_CHARS:
        return JSONResponse({"error": f"text must be 1-{MAX_CHARS} characters"}, 400)
    if not voice.enabled or text not in voice.known:
        return Response(status_code=404)
    if (p := voice.ready(text)):
        return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})
    voice.want(text, urgent=True)
    voice.start()
    return Response(status_code=204)


@mcp.custom_route("/api/stop", methods=["POST"])
async def api_stop(request: Request) -> Response:
    await runner.stop()  # no PIN: anyone in the room may stop the arm
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/do", methods=["POST"])
async def api_do(request: Request) -> Response:
    """The dashboard's buttons go through the same tools Alexa+ calls."""
    if not _pin_ok(request):
        return JSONResponse({"error": "pin"}, 401)
    body = await request.json()
    tool = body.get("tool")
    fn = {"fetch_item": fetch_item, "got_it": got_it, "arm_status": arm_status, "list_items": list_items,
          "medication_today": medication_today, "remember_item": remember_item}.get(tool)
    if not fn:
        return JSONResponse({"error": "unknown tool"}, 400)
    return JSONResponse({"say": await fn(**body.get("args", {}))})


# ------------------------------------------------------------ teach mode ----
# Drive the robot from home to an item with the dashboard's joystick, pose the gripper on it,
# and save: the route and the grip pose are stored under the item's name, then the robot
# retraces the route home on its own.

def _close_segment() -> list[list[float]]:
    now = time.monotonic()
    thr, turn = teach["cmd"]
    if teach["since"] and (abs(thr) >= 0.05 or abs(turn) >= 0.05):
        # in the robot's time: the simulator may run faster than real time in tests
        teach["segments"].append([(now - teach["since"]) * arm.speedup, thr, turn])
    teach["since"] = now
    return [s for s in teach["segments"] if s[0] > 0.02]


async def _set_drive(thr: float, turn: float) -> None:
    if (thr, turn) != teach["cmd"]:
        if teach["item"] is not None:
            _close_segment()
        teach["cmd"] = (thr, turn)
        teach["since"] = time.monotonic()
    await arm._drive(thr, turn)


async def _drive_watchdog() -> None:
    """The dashboard sends the joystick at 10 Hz; if it goes quiet, stop the wheels."""
    while True:
        await asyncio.sleep(0.1)
        if teach["cmd"] != (0.0, 0.0) and time.monotonic() - teach["last"] > 0.4:
            await _set_drive(0.0, 0.0)


def _guard(request: Request) -> Response | None:
    if not _pin_ok(request):
        return JSONResponse({"error": "pin"}, 401)
    if runner.busy():
        return JSONResponse({"error": "busy fetching"}, 409)
    return None


@mcp.custom_route("/api/drive", methods=["POST"])
async def api_drive(request: Request) -> Response:
    if (bad := _guard(request)):
        return bad
    if not arm.has_base:
        return JSONResponse({"error": "this robot has no wheels"}, 400)
    await ensure_arm()
    b = await request.json()
    teach["last"] = time.monotonic()
    await _set_drive(max(-1.0, min(1.0, float(b.get("thr", 0)))), max(-1.0, min(1.0, float(b.get("turn", 0)))))
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/joint", methods=["POST"])
async def api_joint(request: Request) -> Response:
    """Jog the arm: {"pose": [6 numbers]} or {"i": joint, "deg": target}."""
    if (bad := _guard(request)):
        return bad
    await ensure_arm()
    b = await request.json()
    q = arm.q.copy()
    if "pose" in b:
        q[:] = b["pose"]
    else:
        q[int(b["i"])] = float(b["deg"])
    arm.clear_stop()
    try:
        await arm.move(q)
    except Stopped:
        return JSONResponse({"error": "stopped"}, 409)
    return JSONResponse({"q": arm.q.round(1).tolist()})


@mcp.custom_route("/api/teach", methods=["POST"])
async def api_teach(request: Request) -> Response:
    """{"do": "start", "item": "pills"} | {"do": "save", "medication": true} | {"do": "cancel"}
    | {"do": "home"} | {"do": "handover"}   (the last two save the arm's current pose)"""
    if (bad := _guard(request)):
        return bad
    await ensure_arm()
    b = await request.json()
    do = b.get("do")
    if do == "start":
        teach.update(item=str(b.get("item", "")).strip().lower(), segments=[], cmd=(0.0, 0.0), since=time.monotonic())
        return JSONResponse({"teaching": teach["item"]})
    if do == "save" and teach["item"]:
        route = _close_segment()
        p = world.remember(teach["item"], arm.q.tolist(), bool(b.get("medication")), route)
        teach["item"] = None
        _voice_catalog()
        if route and arm.has_base:  # take it home again, the way it came (a job, so nothing else can start meanwhile)
            runner.start_return(route, p.name)
        return JSONResponse({"saved": p.name, "segments": len(route), "pose": p.pose})
    if do == "cancel":
        teach["item"] = None
        await _set_drive(0.0, 0.0)
        return JSONResponse({"ok": True})
    if do in ("home", "handover"):
        setattr(world, do, [round(x, 1) for x in arm.q.tolist()])
        world.save()
        return JSONResponse({do: getattr(world, do)})
    return JSONResponse({"error": "bad request"}, 400)


# ------------------------------------------------------- simulated Alexa+ ----
alexa = AlexaSim(mcp)


@mcp.custom_route("/api/alexa", methods=["POST"])
async def api_alexa(request: Request) -> Response:
    """What the person said -> the simulated Alexa+ picks Reach's tools over MCP -> what it says back."""
    if not _pin_ok(request):
        return JSONResponse({"error": "pin"}, 401)
    b = await request.json()
    text = str(b.get("text", "")).strip()[:300]
    if not text:
        return JSONResponse({"error": "say something"}, 400)
    return JSONResponse(await alexa.ask(text, str(b.get("session", "web"))[:40]))


@mcp.custom_route("/alexa", methods=["GET"])
async def alexa_page(request: Request) -> Response:
    return FileResponse(ROOT / "web" / "alexa.html")


if auth:
    mcp.custom_route("/link", methods=["GET", "POST"])(auth.link_page)


class VoiceStartup:
    """Start rendering Reach's lines on the first ASGI call (uvicorn's lifespan startup), inside the
    running event loop, so prerendering begins as soon as the server is up."""

    def __init__(self, app):
        self.app, self.started = app, False

    async def __call__(self, scope, receive, send):
        if not self.started:
            self.started = True
            _voice_catalog()
        await self.app(scope, receive, send)


def app():
    host = BASE_URL.split("://", 1)[-1].split("/")[0]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[host, f"localhost:{PORT}", f"127.0.0.1:{PORT}"],
        allowed_origins=[BASE_URL, f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
    )
    return VoiceStartup(StripWWWAuthenticate(mcp.streamable_http_app(transport_security=security, host="0.0.0.0")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("REACH on :%d  arm=%s  auth=%s  base=%s", PORT, arm_spec, AUTH_ON, BASE_URL)
    uvicorn.run(app(), host="0.0.0.0", port=PORT)
