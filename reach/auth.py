"""OAuth 2.1 for Alexa+: authorization code + PKCE (S256), refresh tokens.

Alexa+ doesn't do dynamic client registration, so there is one client, set
by environment (the same id/secret go into the add-on's account linking):

    REACH_CLIENT_ID, REACH_CLIENT_SECRET
    REACH_REDIRECT_URIS   comma-separated; Alexa's account-linking redirect URLs
    REACH_OWNER_PIN       the household PIN typed on the consent page

Linking an account shows a small page: "Let Alexa control your REACH arm?"
plus the PIN. That is the whole identity model: one arm, one household.
Tokens are kept in data/tokens.json so a restart doesn't unlink Alexa.
"""

from __future__ import annotations

import hmac
import html
import json
import os
import secrets
import time
from pathlib import Path

from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

SCOPE = "arm"
ACCESS_TTL = 3600
REFRESH_TTL = 90 * 24 * 3600
DEFAULT_REDIRECTS = (
    "https://pitangui.amazon.com/api/skill/link,https://layla.amazon.com/api/skill/link,"
    "https://alexa.amazon.co.jp/api/skill/link,http://localhost:6274/oauth/callback,"
    "http://127.0.0.1:6274/oauth/callback"
)


class ReachAuth:
    def __init__(self, base_url: str, store: Path):
        self.base_url = base_url.rstrip("/")
        self.store = store
        self.client_id = os.getenv("REACH_CLIENT_ID", "reach-alexa")
        self.client_secret = os.getenv("REACH_CLIENT_SECRET") or None
        self.redirects = [u.strip() for u in os.getenv("REACH_REDIRECT_URIS", DEFAULT_REDIRECTS).split(",") if u.strip()]
        self.pin = os.getenv("REACH_OWNER_PIN", "")
        self.pending: dict[str, tuple[str, AuthorizationParams, float]] = {}
        self.codes: dict[str, AuthorizationCode] = {}
        self.access: dict[str, AccessToken] = {}
        self.refresh: dict[str, RefreshToken] = {}
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        if self.store.exists():
            d = json.loads(self.store.read_text())
            now = time.time()
            self.access = {k: AccessToken(**v) for k, v in d.get("access", {}).items() if (v.get("expires_at") or now + 1) > now}
            self.refresh = {k: RefreshToken(**v) for k, v in d.get("refresh", {}).items() if (v.get("expires_at") or now + 1) > now}

    def _save(self):
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({"access": {k: v.model_dump() for k, v in self.access.items()},
                                          "refresh": {k: v.model_dump() for k, v in self.refresh.items()}}))

    # ---- OAuthAuthorizationServerProvider ------------------------------
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        if client_id != self.client_id:
            return None
        return OAuthClientInformationFull(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uris=[AnyUrl(u) for u in self.redirects],
            token_endpoint_auth_method="client_secret_post" if self.client_secret else "none",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=SCOPE,
            client_name="Alexa+",
        )

    async def register_client(self, client_info) -> None:
        raise NotImplementedError("dynamic registration is off")

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        rid = secrets.token_urlsafe(16)
        self.pending[rid] = (client.client_id, params, time.time() + 600)
        return f"{self.base_url}/link?req={rid}"

    async def load_authorization_code(self, client, authorization_code: str) -> AuthorizationCode | None:
        c = self.codes.get(authorization_code)
        return c if c and c.client_id == client.client_id and c.expires_at > time.time() else None

    async def exchange_authorization_code(self, client, authorization_code: AuthorizationCode) -> OAuthToken:
        self.codes.pop(authorization_code.code, None)
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    async def load_refresh_token(self, client, refresh_token: str) -> RefreshToken | None:
        r = self.refresh.get(refresh_token)
        return r if r and r.client_id == client.client_id and (r.expires_at or 0) > time.time() else None

    async def exchange_refresh_token(self, client, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        self.refresh.pop(refresh_token.token, None)  # rotate
        return self._issue(client.client_id, scopes or refresh_token.scopes, refresh_token.resource)

    async def load_access_token(self, token: str) -> AccessToken | None:
        a = self.access.get(token)
        if a and (a.expires_at or 0) > time.time():
            return a
        return None

    async def revoke_token(self, token) -> None:
        self.access.pop(token.token, None)
        self.refresh.pop(token.token, None)
        self._save()

    def _issue(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        now = int(time.time())
        at, rt = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.access[at] = AccessToken(token=at, client_id=client_id, scopes=scopes, expires_at=now + ACCESS_TTL,
                                      resource=resource, subject="household")
        self.refresh[rt] = RefreshToken(token=rt, client_id=client_id, scopes=scopes, expires_at=now + REFRESH_TTL,
                                        resource=resource, subject="household")
        self._save()
        return OAuthToken(access_token=at, expires_in=ACCESS_TTL, scope=" ".join(scopes), refresh_token=rt)

    # ---- the consent page ----------------------------------------------
    async def link_page(self, request: Request) -> Response:
        rid = request.query_params.get("req") or (await request.form()).get("req", "")
        item = self.pending.get(rid)
        if not item or item[2] < time.time():
            return HTMLResponse(_page("This link has expired. Start account linking again in the Alexa app."), 400)
        if request.method == "GET":
            return HTMLResponse(_page(_FORM.format(req=html.escape(rid), err="")))
        form = await request.form()
        if not self.pin or not hmac.compare_digest(str(form.get("pin", "")), self.pin):
            return HTMLResponse(_page(_FORM.format(req=html.escape(rid), err="<p class=err>That PIN didn't match.</p>")), 401)
        client_id, params, _ = self.pending.pop(rid)
        code = secrets.token_urlsafe(24)
        self.codes[code] = AuthorizationCode(
            code=code, scopes=params.scopes or [SCOPE], expires_at=time.time() + 300, client_id=client_id,
            code_challenge=params.code_challenge, redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly, resource=params.resource,
            subject="household")
        return RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state), 302)


_FORM = """<h1>Let Alexa control your REACH arm?</h1>
<p>Alexa will be able to ask the arm to fetch, hold out and put back the items it knows,
and to stop it. It can't change where things are without you.</p>
{err}
<form method=post><input type=hidden name=req value="{req}">
<label>Household PIN <input name=pin type=password inputmode=numeric autocomplete=off autofocus required></label>
<button>Allow</button></form>"""


def _page(body: str) -> str:
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>REACH · link Alexa</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:440px;margin:12vh auto;padding:0 16px;color:#1d2125;background:#f6f4ef}}
h1{{font-size:22px}}label{{display:grid;gap:6px;margin:18px 0}}input{{font:inherit;padding:10px;border:1px solid #c9c4b8;border-radius:8px}}
button{{font:inherit;padding:10px 18px;border:0;border-radius:8px;background:#1f6f5c;color:#fff;cursor:pointer}}.err{{color:#b3261e}}</style>
{body}"""


class StripWWWAuthenticate:
    """Alexa+ wants a bare 401: no WWW-Authenticate header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def _send(msg):
            if msg["type"] == "http.response.start" and msg.get("status") == 401:
                msg = dict(msg)
                msg["headers"] = [(k, v) for k, v in msg.get("headers", []) if k.lower() != b"www-authenticate"]
            await send(msg)

        await self.app(scope, receive, _send)
