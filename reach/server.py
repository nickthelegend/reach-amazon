"""REACH: an Alexa+ MCP server that lets someone with limited mobility ask a
robot arm to bring them things.

    "Alexa, ask Reach to bring me my pills."
    "Alexa, tell Reach I've got it."
    "Alexa, tell Reach to stop."
    "Alexa, ask Reach if I took my pills today."

Run:   python -m reach.server            (serves /mcp, OAuth, and the dashboard on :8790)
Env:   REACH_ARM=sim | /dev/cu.usbserial-XXXX     which arm to drive (default sim)
       REACH_BASE_URL=https://<public-url>        what Alexa+ reaches (tunnel or host)
       REACH_AUTH=0                               no OAuth (local testing with MCP Inspector)
       REACH_OWNER_PIN, REACH_CLIENT_ID, REACH_CLIENT_SECRET, REACH_REDIRECT_URIS   (auth.py)
       THENAR_ARMS_DIR                            path to thenar-arms, for the SO-101 table guard
"""

from __future__ import annotations

import asyncio
import hmac
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

from .arm import SimArm, SO101Arm, load_guard
from .auth import SCOPE, ReachAuth, StripWWWAuthenticate
from .jobs import Runner
from .world import World

log = logging.getLogger("reach")
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("REACH_DATA", ROOT / "data"))
PORT = int(os.getenv("PORT", "8790"))
BASE_URL = os.getenv("REACH_BASE_URL", f"http://localhost:{PORT}").rstrip("/")
AUTH_ON = os.getenv("REACH_AUTH", "1") != "0"

world = World(DATA / "world.json")
arm_spec = os.getenv("REACH_ARM", "sim")
arm = SimArm() if arm_spec == "sim" else SO101Arm(arm_spec, load_guard(os.getenv("THENAR_ARMS_DIR")))
arm.speedup = float(os.getenv("REACH_SIM_SPEEDUP", "1")) if arm_spec == "sim" else 1.0
runner = Runner(arm, world)
_arm_ready: asyncio.Task | None = None


async def ensure_arm() -> None:
    """Connect the arm once, on first use (the SO-101 takes a few seconds to arm)."""
    global _arm_ready
    if _arm_ready is None:
        _arm_ready = asyncio.create_task(arm.connect())
    await _arm_ready


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


def _items_sentence() -> str:
    names = sorted(world.places)
    return ", ".join(names[:-1]) + (" and " + names[-1] if len(names) > 1 else (names[0] if names else "nothing yet"))


@mcp.tool(annotations=ACT)
async def fetch_item(item: str) -> str:
    """Bring the person an item. The arm picks it up, holds it out toward them, and waits until they say
    they have it. Returns immediately while the arm moves (about 15 seconds).

    item: what they asked for, in their words, e.g. "my pills", "water", "reading glasses", "the remote".
    """
    place = world.find(item)
    if not place:
        return f"I don't know where the {item} is. I can bring your {_items_sentence()}."
    if runner.busy():
        return f"I'm still busy with the {runner.job.item}. Say stop if you want me to stop."
    await ensure_arm()
    runner.start_fetch(place)
    return f"On my way to get your {place.name}. I'll hold it out in about fifteen seconds. Tell me when you've got it."


@mcp.tool(annotations=ACT)
async def got_it() -> str:
    """The person has taken the item the arm is holding out. Opens the gripper and sends the arm home."""
    if runner.taken():
        return "Okay, letting go."
    if runner.busy():
        return f"I'm still {runner.job.stage}. I'll tell you when it's ready to take."
    return "I'm not holding anything right now."


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def stop_arm() -> str:
    """Stop the arm right now, wherever it is. Use whenever the person says stop, wait, or sounds alarmed."""
    await runner.stop()
    return "Stopped."


@mcp.tool(annotations=READ)
async def arm_status() -> str:
    """What the arm is doing right now, e.g. still bringing the pills, or holding them out."""
    if not runner.job:
        return "I'm resting and ready."
    j = runner.job
    if runner.busy():
        return f"I'm {j.stage}." if j.stage != "holding it out" else f"I'm holding out your {j.item}. Say you've got it when you take it."
    return {"taken": f"I brought your {j.item}.", "not_taken": f"Nobody took the {j.item}, so I put it back.",
            "stopped": "I stopped.", "failed": f"Something went wrong getting the {j.item}: {j.detail}."}.get(j.outcome or "", "I'm ready.")


@mcp.tool(annotations=READ)
async def list_items() -> str:
    """The items the arm knows how to bring."""
    return f"I can bring your {_items_sentence()}."


@mcp.tool(annotations=READ)
async def medication_today() -> str:
    """Whether the person has been handed their medication today, and when. For questions like
    "did I take my pills?"."""
    taken = world.medication_today()
    if not taken:
        return "I haven't handed you any medication today."
    times = [time.strftime("%-I:%M %p", time.localtime(e["t"])) for e in taken]
    items = sorted({e["item"] for e in taken})
    return f"I handed you your {' and '.join(items)} today at {', '.join(times)}."


@mcp.tool(annotations=ACT)
async def remember_item(name: str, is_medication: bool = False) -> str:
    """Teach the arm where something is: the arm (or the leader arm) has been placed on the item by a
    helper, and this saves that spot under the given name. Only when the person or a helper asks to.

    name: what to call it, e.g. "blood pressure pills". is_medication: track it in the medication log.
    """
    if runner.busy():
        return "I can't learn a spot while I'm moving."
    p = world.remember(name, arm.q.tolist(), is_medication)
    return f"Got it. I'll remember where your {p.name} {'are' if p.name.endswith('s') else 'is'}."


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
        "items": {k: {"medication": p.medication, "pose": p.pose} for k, p in world.places.items()},
        "medication_today": world.medication_today(),
        "log": world.log[-12:],
        "auth": AUTH_ON,
    })


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


if auth:
    mcp.custom_route("/link", methods=["GET", "POST"])(auth.link_page)


def app():
    host = BASE_URL.split("://", 1)[-1].split("/")[0]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[host, f"localhost:{PORT}", f"127.0.0.1:{PORT}"],
        allowed_origins=[BASE_URL, f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
    )
    return StripWWWAuthenticate(mcp.streamable_http_app(transport_security=security, host="0.0.0.0"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("REACH on :%d  arm=%s  auth=%s  base=%s", PORT, arm_spec, AUTH_ON, BASE_URL)
    uvicorn.run(app(), host="0.0.0.0", port=PORT)
