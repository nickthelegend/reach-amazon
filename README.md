# Reach

**"Alexa, ask Reach to bring me my pills."** A small robot drives to where the
pills are, picks them up, comes back and holds them out to someone who can't
easily get up, then lets go when they say they have it.

Assistive robot arms cost tens of thousands of dollars and need their own app or
joystick. Reach runs on the **Thenar Walker**, a 3-D-printed four-wheel robot
carrying an SO-101 arm (built for Techfest RoboReach,
[nickthelegend/thenar-walker](https://github.com/nickthelegend/thenar-walker)),
and on the voice assistant people already have.

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

## Teach once, fetch by voice

There's no camera and no map. A helper teaches each place once from the dashboard's
**Teach a place** tab:
1. Start at the robot's home spot beside the person.
2. Drive to the item with the joystick.
3. Close the gripper on it with the sliders, then press save.

Reach stores the route and the grip, and the robot drives itself home. A fetch then:
1. replays the route
2. hovers above the item and grips it (the hover moves the arm toward its folded home pose, so it
   works on any robot without knowing joint directions)
3. retraces the route in reverse
4. holds the item out

Routes are open loop, so keep them short (a few metres on a flat floor).

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
- **The Thenar Walker** (`reach/walker.py`). It speaks the robot firmware's WebSocket protocol on
  the robot's own Wi-Fi (`ws://192.168.4.1:81/`):
  - Joints switch on one at a time, within the robot's calibrated limits.
  - A 40 ms drive heartbeat keeps the firmware's 300 ms failsafe meaningful: if REACH dies, the
    robot stops.
  - Stops and drive changes go out immediately.

  `reach/mock_walker.py` mirrors the firmware for tests.
- **Simulated Alexa+** (`reach/alexa_sim.py`, page `/alexa`). This is the hackathon's simulated
  Alexa+ path.
  - It's a real MCP client to this server that discovers the tools and lets a model choose them:
    AWS Bedrock (Converse tool use), any OpenAI-compatible model, or offline rules.
  - Browser speech in and out.
  - Every tool call is shown with its latency.
- **Taught places, not hard-coded coordinates** (`reach/world.py`). The starter
  layout was solved with the arm's inverse kinematics. Real places are taught by
  putting the arm on the item and saying "remember…".

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
REACH_AUTH=0 .venv/bin/python -m reach.server        # simulated arm, http://localhost:8790
.venv/bin/python -m pytest -q                        # real server + MCP client + OAuth flow
```

With the Thenar Walker (join its `<Team>_TF` Wi-Fi first; keep internet through Ethernet or a
phone tether for the model):

```bash
REACH_ARM=walker .venv/bin/python -m reach.server          # dashboard :8790, simulator :8790/alexa
```

Simulated Alexa+ with Bedrock: set `AWS_PROFILE` (or keys), `AWS_REGION`, and
`REACH_BEDROCK_MODEL`. With any OpenAI-compatible model: set `REACH_LLM_BASE_URL`,
`REACH_LLM_API_KEY` and `REACH_LLM_MODEL`.

With a bench SO-101 follower on USB:

```bash
REACH_ARM=/dev/cu.usbserial-XXXX THENAR_ARMS_DIR=~/thenar-arms .venv/bin/python -m reach.server
```

## Connect it to real Alexa+

Alexa+ add-ons are in private preview: request access, and you'll receive the Alexa AI CLI
(`@alexa-ai/cli`, not public on npm).

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

- **Tested:** 9 automated tests.
  - the MCP tools over Streamable HTTP and the OAuth PKCE flow
  - the Walker driver against the mock firmware
  - exact route reversal
  - teach → voice fetch → handover → medication log
  - stop always winning
- **Checked in the browser:** the dashboard and the simulator.
- **Not yet done:** running on the physical Walker, and real Alexa+, which is gated by its
  preview.

## License

MIT
