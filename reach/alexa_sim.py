"""A simulated Alexa+ for the web demo (the hackathon's "simulated Alexa+ experience" path).

It behaves the way Alexa+ does with an add-on: it is an MCP client to the Reach server,
discovers the tools, lets a language model decide which to call from what the person said,
calls them over MCP, and answers in one or two spoken sentences. The tool calls are the same
JSON-RPC requests real Alexa+ would send; only the speech recognition and the model differ.

Engines, first one configured wins:
  bedrock   AWS Bedrock Converse with tool use (boto3; AWS credentials + REACH_BEDROCK_MODEL)
  openai    any OpenAI-compatible endpoint (REACH_LLM_BASE_URL, REACH_LLM_API_KEY, REACH_LLM_MODEL)
  rules     offline intent matching, so the demo never depends on a network
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import httpx

from mcp import Client

log = logging.getLogger("reach.alexa")

SYSTEM = (
    "You are Alexa+. The person has enabled the Reach add-on: a robot that fetches things for them because "
    "they can't easily get up. Use Reach's tools to do what they ask. If they say stop, wait, or sound alarmed, "
    "call stop_arm before anything else. When they say they have the item, call got_it. Answer in one or two "
    "short, warm, spoken sentences, no lists or markdown. Don't invent items Reach doesn't know."
)
MAX_ROUNDS = 4


def engine_name() -> str:
    if os.getenv("REACH_BEDROCK_MODEL") and (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")):
        return "bedrock"
    if os.getenv("REACH_LLM_MODEL") and os.getenv("REACH_LLM_BASE_URL"):
        return "openai"
    return "rules"


class AlexaSim:
    def __init__(self, server):
        self.server = server  # the MCPServer; connected in-process over the MCP protocol
        self.sessions: dict[str, list] = {}
        self.engine = engine_name()

    async def ask(self, text: str, session: str = "default") -> dict:
        calls: list[dict] = []
        async with Client(self.server) as mcp:
            tools = (await mcp.list_tools()).tools

            async def call(name: str, args: dict) -> str:
                t0 = time.perf_counter()
                r = await mcp.call_tool(name, args)
                out = r.content[0].text if r.content else ""
                calls.append({"tool": name, "args": args, "ms": round((time.perf_counter() - t0) * 1000, 1), "result": out})
                return out

            try:
                if self.engine == "bedrock":
                    reply = await self._bedrock(text, session, tools, call)
                elif self.engine == "openai":
                    reply = await self._openai(text, session, tools, call)
                else:
                    reply = await self._rules(text, call)
            except Exception as e:  # noqa: BLE001 - fall back rather than leave the person unanswered
                log.warning("%s engine failed (%s); using rules", self.engine, e)
                reply = await self._rules(text, call)
        return {"heard": text, "reply": reply, "calls": calls, "engine": self.engine}

    # ------------------------------------------------------------ rules ----
    async def _rules(self, text: str, call) -> str:
        s = text.lower().strip()
        if re.search(r"\b(stop|wait|halt|no no)\b", s):
            return await call("stop_arm", {})
        if re.search(r"(got it|have it|thank|let go|release|took it)", s):
            return await call("got_it", {})
        if re.search(r"(did i|have i).*(pill|med|tablet)", s):
            return await call("medication_today", {})
        if re.search(r"(what can you|what do you know|what things|list)", s):
            return await call("list_items", {})
        if re.search(r"(what are you doing|status|where are you|how long)", s):
            return await call("arm_status", {})
        m = re.search(r"(?:bring|get|fetch|pass|hand|give|grab)(?: me)?(?: the| my)?\s+(.+?)(?: please| now|\?|$)", s)
        if m:
            return await call("fetch_item", {"item": m.group(1).strip()})
        return "Sorry, I didn't catch that. You can say, bring me my pills."

    # ------------------------------------------------------- openai-style ----
    async def _openai(self, text: str, session: str, tools, call) -> str:
        msgs = self.sessions.setdefault(session, [])[-8:]
        msgs = [{"role": "system", "content": SYSTEM}, *msgs, {"role": "user", "content": text}]
        spec = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                  "parameters": t.input_schema}} for t in tools]
        base = os.environ["REACH_LLM_BASE_URL"].rstrip("/")
        headers = {"Authorization": f"Bearer {os.getenv('REACH_LLM_API_KEY', '')}"}
        async with httpx.AsyncClient(timeout=30) as c:
            for _ in range(MAX_ROUNDS):
                r = await c.post(f"{base}/chat/completions", headers=headers,
                                 json={"model": os.environ["REACH_LLM_MODEL"], "messages": msgs, "tools": spec,
                                       "temperature": 0.2, "max_tokens": 300})
                r.raise_for_status()
                m = r.json()["choices"][0]["message"]
                msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
                if not m.get("tool_calls"):
                    break
                for tc in m["tool_calls"]:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    out = await call(tc["function"]["name"], args)
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        reply = (msgs[-1].get("content") or "").strip() if msgs[-1]["role"] == "assistant" else ""
        self.sessions[session] = [m for m in msgs if m["role"] in ("user", "assistant") and m.get("content")][-8:]
        return reply or (msgs[-1].get("content") if msgs[-1]["role"] == "tool" else "Okay.")

    # ------------------------------------------------------------ bedrock ----
    async def _bedrock(self, text: str, session: str, tools, call) -> str:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        history = self.sessions.setdefault(session, [])[-8:]
        msgs = [*history, {"role": "user", "content": [{"text": text}]}]
        cfg = {"tools": [{"toolSpec": {"name": t.name, "description": t.description or t.name,
                                       "inputSchema": {"json": t.input_schema}}} for t in tools]}
        reply = ""
        for _ in range(MAX_ROUNDS):
            r = await asyncio.to_thread(client.converse, modelId=os.environ["REACH_BEDROCK_MODEL"], messages=msgs,
                                        system=[{"text": SYSTEM}], toolConfig=cfg,
                                        inferenceConfig={"maxTokens": 300, "temperature": 0.2})
            out = r["output"]["message"]
            msgs.append(out)
            uses = [b["toolUse"] for b in out["content"] if "toolUse" in b]
            reply = " ".join(b["text"] for b in out["content"] if "text" in b).strip()
            if r.get("stopReason") != "tool_use" or not uses:
                break
            results = []
            for u in uses:
                res = await call(u["name"], u.get("input") or {})
                results.append({"toolResult": {"toolUseId": u["toolUseId"], "content": [{"text": res}]}})
            msgs.append({"role": "user", "content": results})
        self.sessions[session] = [m for m in msgs if all("text" in b for b in m["content"])][-8:]
        return reply or "Okay."
