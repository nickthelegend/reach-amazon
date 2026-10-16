# Reach

**"Alexa, ask Reach to bring me my pills."** A robot arm next to someone who
can't easily get up picks the item up, holds it out to them, and lets go
when they say they have it.

Assistive robot arms exist, but they cost tens of thousands of dollars and need
their own app or joystick. Reach uses a 3-D-printable SO-101 arm and the voice
assistant people already have.

Built for **Build, Ship, Shape: Amazon Developer Hackathon**, Alexa+ track
(+ AWS Builder mini challenge).

## What you can say

| You say | What happens |
|---|---|
| "bring me my pills" / "get my water" / "pass me the remote" | fetch: reach, pick up, bring it over, hold it out |
| "I've got it" / "thanks" | let go, go home; medication handovers are logged |
| "stop" / "wait" | stops at once, at any stage, including while holding something out |
| "did I take my pills today?" | "I handed you your pills today at 8:14 AM" |
| "what can you bring me?" | the items it knows |
| "remember this is my blood pressure pills" | a helper put the arm on the item; it saves the spot |

## How it works

```
Alexa+ ──(MCP 2025-11-25, Streamable HTTP, OAuth 2.1 + PKCE)──► Reach server ──► arm
                                                                   │            (sim, or SO-101 over USB)
                                                     dashboard ◄───┘  big STOP, 3-D view, medication log
```

- **MCP server** (`reach/server.py`). The tools are:
  - `fetch_item`, `got_it` and `stop_arm` act on the arm.
  - `arm_status`, `list_items` and `medication_today` read state.
  - `remember_item` saves a new spot.

  Alexa+ wants answers in under 500 ms, so a fetch starts and returns at once
  while the arm keeps moving. Alexa+ asks `arm_status` for progress.
- **Safety.**
  - Motion is interpolated at 50 Hz with per-joint speed caps.
  - `stop_arm` (and the dashboard's STOP, which needs no PIN) wins at every stage.
  - On the SO-101, every streamed pose passes the thenar-arms table guard. The
    firmware must acknowledge `STOP` before the arm is armed.
- **Account linking** (`reach/auth.py`). OAuth 2.1 authorization code with PKCE
  (S256) and rotating refresh tokens. A household-PIN consent page confirms the
  link, and an unauthenticated call gets a bare 401 with no `WWW-Authenticate`
  header, as Alexa+ requires.
- **Taught places, not hard-coded coordinates** (`reach/world.py`). The starter
  layout was solved with the arm's inverse kinematics. Real places are taught by
  putting the arm on the item and saying "remember…".

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
REACH_AUTH=0 .venv/bin/python -m reach.server        # simulated arm, http://localhost:8790
.venv/bin/python -m pytest -q                        # real server + MCP client + OAuth flow
```

With the SO-101 follower on USB:

```bash
REACH_ARM=/dev/cu.usbserial-XXXX THENAR_ARMS_DIR=~/thenar-arms .venv/bin/python -m reach.server
```

## Connect it to Alexa+

1. Expose the server over HTTPS, e.g.
   `cloudflared tunnel --url http://localhost:8790`. Set `REACH_BASE_URL` to that
   URL.
2. Set `REACH_OWNER_PIN`, `REACH_CLIENT_ID` and `REACH_CLIENT_SECRET`, and set
   `REACH_REDIRECT_URIS` to the account-linking redirect URLs Alexa shows you.
3. Install the Alexa AI CLI, then run `alexa-ai configure`.
4. Create the add-on:
   `alexa-ai new mcp --name "Reach" --locale en-US --mcp-server-url "$REACH_BASE_URL/mcp"`.
5. Fill in `addon-package/addon.json` (description, example phrases, icons,
   account linking with the client id/secret), then run `alexa-ai deploy`.
6. Test in the Alexa+ web simulator: "ask Reach to bring me my water".

## Status

- **Tested:** the server, the tools, stop, the medication log, and the OAuth flow
  (automated tests), plus the dashboard with the simulated arm.
- **Not yet done:** deploying to Alexa+ and running on a physical SO-101.

## License

MIT
