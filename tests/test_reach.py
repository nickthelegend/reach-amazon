"""REACH end to end: a real server process, a real MCP client over Streamable HTTP,
the simulated arm, and the full OAuth 2.1 + PKCE linking flow Alexa+ uses."""

import asyncio
import base64
import hashlib
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def server(tmp_path, request):
    auth = getattr(request, "param", False)
    port = free_port()
    env = {**os.environ, "PORT": str(port), "REACH_DATA": str(tmp_path), "REACH_ARM": "sim",
           "REACH_SIM_SPEEDUP": "25", "REACH_AUTH": "1" if auth else "0", "REACH_OWNER_PIN": "2468",
           "REACH_CLIENT_ID": "alexa-test", "REACH_CLIENT_SECRET": "s3cret",
           "REACH_REDIRECT_URIS": "https://layla.amazon.com/api/skill/link/TEST",
           "REACH_BASE_URL": f"http://127.0.0.1:{port}"}
    proc = subprocess.Popen([sys.executable, "-m", "reach.server"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(base + "/api/state", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(proc.stdout.read().decode())
    yield base
    proc.terminate()
    proc.wait(5)


async def call(client, tool, **args) -> str:
    r = await client.call_tool(tool, args)
    return r.content[0].text


def test_tools_listed(server):
    async def go():
        async with Client(server + "/mcp") as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"fetch_item", "got_it", "stop_arm", "arm_status", "list_items", "medication_today",
                    "remember_item"} <= names
    asyncio.run(go())


def test_fetch_pills_handover_and_medication_log(server):
    async def go():
        async with Client(server + "/mcp") as c:
            assert "haven't" in await call(c, "medication_today")
            t0 = time.monotonic()
            say = await call(c, "fetch_item", item="my blood pressure pills")
            assert "pills" in say and time.monotonic() - t0 < 0.5  # Alexa+ wants < 500 ms
            for _ in range(200):
                if "holding out" in await call(c, "arm_status"):
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("arm never held the pills out")
            assert "letting go" in (await call(c, "got_it")).lower()
            for _ in range(200):
                if "brought" in await call(c, "arm_status"):
                    break
                await asyncio.sleep(0.1)
            assert "pills" in await call(c, "medication_today")
    asyncio.run(go())


def test_unknown_item_and_stop(server):
    async def go():
        async with Client(server + "/mcp") as c:
            assert "don't know" in await call(c, "fetch_item", item="umbrella")
            await call(c, "fetch_item", item="water")
            await asyncio.sleep(0.3)
            assert await call(c, "stop_arm") == "Stopped."
            await asyncio.sleep(0.3)
            assert "stopped" in (await call(c, "arm_status")).lower()
    asyncio.run(go())


@pytest.mark.parametrize("server", [True], indirect=True)
def test_oauth_pkce_link_then_call(server):
    h = httpx.Client(base_url=server, follow_redirects=False, timeout=5)
    # 1. unauthenticated MCP call: bare 401, no WWW-Authenticate (Alexa+ requirement)
    r = h.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
               headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 401 and "www-authenticate" not in r.headers
    # 2. discovery
    meta = h.get("/.well-known/oauth-authorization-server").json()
    assert "S256" in meta["code_challenge_methods_supported"]
    assert h.get("/.well-known/oauth-protected-resource/mcp").status_code == 200
    # 3. authorize with PKCE -> consent page -> PIN -> code
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    redirect = "https://layla.amazon.com/api/skill/link/TEST"
    r = h.get("/authorize", params={"response_type": "code", "client_id": "alexa-test", "redirect_uri": redirect,
                                     "code_challenge": challenge, "code_challenge_method": "S256", "state": "xyz",
                                     "resource": server + "/mcp"})
    assert r.status_code in (302, 307)
    link = urlparse(r.headers["location"])
    req = parse_qs(link.query)["req"][0]
    assert h.post("/link", data={"req": req, "pin": "0000"}).status_code == 401
    r = h.post("/link", data={"req": req, "pin": "2468"})
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["state"] == ["xyz"]
    # 4. token exchange
    tok = h.post("/token", data={"grant_type": "authorization_code", "code": q["code"][0], "redirect_uri": redirect,
                                 "client_id": "alexa-test", "client_secret": "s3cret", "code_verifier": verifier,
                                 "resource": server + "/mcp"}).json()
    assert tok["token_type"] == "Bearer" and tok["refresh_token"]
    # 5. the token works on /mcp
    r = h.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                                        "clientInfo": {"name": "t", "version": "1"}}},
               headers={"Accept": "application/json, text/event-stream", "Authorization": f"Bearer {tok['access_token']}"})
    assert r.status_code == 200, r.text
    # 6. refresh rotates the token
    tok2 = h.post("/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                                  "client_id": "alexa-test", "client_secret": "s3cret"}).json()
    assert tok2["access_token"] != tok["access_token"]
