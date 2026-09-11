"""LLM providers: mock (offline), Gemini, OpenAI-compatible.

The brain works with **zero configuration** using the `mock` provider —
extraction is done by the rule-based insight engine, and the "teacher"
replies come from templates. Drop in a real API key (Gemini or OpenAI)
and the same endpoints use the LLM for richer extraction and replies.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

import httpx

from .generator import CANONICAL, KNOWN_RULE_IDS, summarize_rules

_SCHEMA_EXAMPLE = json.dumps(
    {
        "name": "short strategy name",
        "timeframe": "chart timeframe in minutes as string, e.g. '15'",
        "market": "crypto | forex | stocks | futures | any",
        "side": "long | short | both",
        "summary": "2-3 sentence plain-English description of the rules",
        "notes": ["anything ambiguous or not representable in the schema"],
        "stop": {"type": "atr|percent|swing", "params": {"length": 14, "mult": 2.0}},
        "target": {"type": "rr|percent|opposite", "params": {"rr": 2.0}},
        "rules": {
            "entry": [{"id": "...", "params": {}}],
            "exit": [{"id": "...", "params": {}}],
            "short_entry": [{"id": "...", "params": {}}],
            "short_exit": [{"id": "...", "params": {}}],
            "filters": [{"id": "...", "params": {}}],
            "short_filters": [{"id": "...", "params": {}}],
        },
    },
    indent=2,
)

SYSTEM_PROMPT = (
    "You are the brain of a TradingView indicator. The user shows you\n"
    "trading-strategy videos (YouTube) and you convert what is taught in them into a\n"
    "strategy configuration for an automated Pine Script indicator.\n\n"
    "The strategy schema you must return as JSON:\n"
    + _SCHEMA_EXAMPLE
    + "\n\nOnly these rule ids exist (never invent new ones):\n"
    + json.dumps(sorted(KNOWN_RULE_IDS))
    + "\n\nParams you may use per rule id (leave params empty to use defaults):\n"
    "  ema_cross_up/down, sma_cross_up/down        : {\"fast\": n, \"slow\": n}\n"
    "  macd_cross_up/down                          : {\"fast\": 12, \"slow\": 26, \"smooth\": 9}\n"
    "  rsi_cross_up/down                           : {\"length\": 14, \"level\": n}   (crossover/crossunder a level)\n"
    "  bb_lower_touch / bb_upper_touch             : {\"length\": 20, \"mult\": 2.0}\n"
    "  supertrend_up/down                          : {\"factor\": 3.0, \"atr\": 10}\n"
    "  breakout_high / breakdown_low               : {\"lookback\": n}  (highest high / lowest low)\n"
    "  support_bounce / resistance_reject          : {\"left\": 10, \"right\": 10}   (pivot bars)\n"
    "  ichimoku_cloud_up / ichimoku_cloud_down     : {\"conversion\": 9, \"base\": 26, \"lagging\": 52, \"disp\": 26}\n"
    "  ichimoku_above_cloud / ichimoku_below_cloud : same params\n"
    "  ema_stack_bull / ema_stack_bear             : {\"fast\": 10, \"mid\": 20, \"slow\": 50}\n"
    "  rsi_below / rsi_above                       : {\"length\": 14, \"level\": n}\n"
    "  price_above_ema / price_below_ema           : {\"length\": 200}\n"
    "  vwap_above / vwap_below                     : {}\n"
    "  adx_above                                   : {\"length\": 14, \"level\": 25}\n"
    "  volume_spike                                : {\"length\": 20, \"mult\": 2.0}\n"
    "  session                                     : {\"tz\": \"America/New_York\", \"start\": 540, \"end\": 1020}  (minutes)\n"
    "\nGuidelines:\n"
    "- entry/exit are long-side signals that happen at a moment in time (crossovers,\n"
    "  touches, breakouts); short_entry/short_exit are the same but for short trades.\n"
    "- filters are ongoing conditions (RSI below X, price above EMA, session) for the\n"
    "  long side; short_filters for the short side.\n"
    "- If the video teaches shorting, populate the short_* buckets and set side to\n"
    "  'both' (or 'short' if only shorts). If a short strategy is described as the\n"
    "  mirror of the long one, mirror the rule ids accordingly.\n"
    "- If the video mentions moving-average crossovers or RSI levels, prefer those ids.\n"
    "- Return ONLY the JSON object, no markdown fences, no commentary.\n"
)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLM(ABC):
    @abstractmethod
    async def extract(self, transcript: str, video: dict) -> dict:
        """Transcript → strategy dict (may be partial)."""

    @abstractmethod
    async def reply(self, user_message: str, context: str) -> str:
        """Free-form chat reply from the brain."""

    @property
    @abstractmethod
    def label(self) -> str:
        ...


class MockLLM(LLM):
    """Offline provider: extraction via heuristics, replies via templates."""

    label = "mock (offline)"

    async def extract(self, transcript: str, video: dict) -> dict:
        from .insight import extract_heuristic

        return extract_heuristic(transcript)

    async def reply(self, user_message: str, context: str) -> str:
        from .insight import detect_intent, INTENTS

        intent = detect_intent(user_message)
        name = INTENTS[intent][0]
        if intent in ("help",):
            return (
                "Here's what I can do:\n\n"
                "🎥 **Teach me a strategy** — paste any YouTube link and I'll watch the video "
                "(captions), pull out the trading rules, and generate your indicator.\n\n"
                "📊 **Indicator** — the current strategy is always compiled to Pine Script v6 in "
                "`pine/indicator.pine` (signal arrows + stop/target levels) with alert webhooks "
                "back to me, and `pine/strategy.pine` for backtesting.\n\n"
                "🧠 **Learning** — every BUY/SELL alert your TradingView indicator fires comes back "
                "to me via webhook. I check win rate after the exit signal and suggest improvements.\n\n"
                "🔌 **Real AI** — install a Gemini or OpenAI key in Settings (top right) and I'll "
                "extract strategies with far deeper understanding.\n\n"
                "Try it now: paste a video link, or click **“Feed me a demo video”**."
            )
        if intent in ("status",):
            return f"{name} — here's the current brain state:\n\n{context}"
        if intent in ("learn", "clear", "save"):
            return (
                f"{name} — I'm running in offline (mock) mode, so I use the built-in rule engine "
                f"instead of an LLM. Install a Gemini or OpenAI key in **Settings** to unlock the "
                f"full chat brain. The strategy generation itself already works — paste a video link!"
            )
        if intent == "greet":
            return "👋 Hey! I'm your TradingView brain. Paste a YouTube video link and I'll learn the strategy from it."
        if intent == "thanks":
            return "You're welcome! Paste a video link any time — the more I watch, the more I learn."
        return (
            f"{name} — I'm running in offline (mock) mode. The most useful thing you can do right "
            f"now is paste a **YouTube link** and I'll extract the strategy with my rule engine. "
            f"Install a Gemini or OpenAI key in Settings for full chat understanding."
        )


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str = "", native_video: bool = True):
        self.api_key = api_key
        self.model = model or "gemini-2.0-flash"
        self.native_video = native_video

    @property
    def label(self):
        return f"gemini ({self.model})"

    async def _chat(self, contents: list[dict], system: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    async def extract(self, transcript: str, video: dict) -> dict:
        from .insight import extract_heuristic

        if self.native_video and video.get("url"):
            contents = [{
                "role": "user",
                "parts": [
                    {"text": "Extract the trading strategy taught in this video (use its captions/audio)."},
                    {"file_data": {"file_uri": video["url"], "mime_type": "video/youtube"}},
                ],
            }]
            try:
                raw = await self._chat(contents, SYSTEM_PROMPT)
                parsed = _extract_json(raw)
                if parsed:
                    return parsed
            except Exception:
                pass  # fall back to transcript
        contents = [{
            "role": "user",
            "parts": [
                {"text": "Extract the trading strategy from this video transcript."},
                {"text": transcript[-60000:]},
            ],
        }]
        raw = await self._chat(contents, SYSTEM_PROMPT)
        return _extract_json(raw) or extract_heuristic(transcript)

    async def reply(self, user_message: str, context: str) -> str:
        contents = [{
            "role": "user",
            "parts": [{"text": f"{user_message}\n\nCurrent brain state:\n{context}"}],
        }]
        return await self._chat(
            contents,
            "You are a friendly trading-bot brain. Keep replies short and concrete. "
            "You can explain the current strategy, and remind the user they can paste a "
            "YouTube link to teach you a new one.",
        )


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"

    @property
    def label(self):
        return f"openai ({self.model})"

    async def _chat(self, messages: list[dict]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                url,
                json={"model": self.model, "messages": messages},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def extract(self, transcript: str, video: dict) -> dict:
        from .insight import extract_heuristic

        raw = await self._chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Video transcript:\n\n{transcript[-60000:]}"},
        ])
        return _extract_json(raw) or extract_heuristic(transcript)

    async def reply(self, user_message: str, context: str) -> str:
        return await self._chat([
            {"role": "system", "content": "You are a friendly trading-bot brain. Keep replies short and concrete."},
            {"role": "user", "content": f"{user_message}\n\nCurrent brain state:\n{context}"},
        ])


def get_llm(settings: dict) -> LLM:
    provider = (settings.get("provider") or "mock").lower()
    key = (settings.get("api_key") or "").strip()
    if provider == "gemini" and key:
        return GeminiLLM(key, settings.get("model", ""), settings.get("native_video", True))
    if provider == "openai" and key:
        return OpenAILLM(key, settings.get("model", ""))
    return MockLLM()
