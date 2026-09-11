"""Trading Brain — the brain behind a TradingView indicator.

Run:  .venv/bin/uvicorn brain.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from brain.core import generator, learn
from brain.core.insight import INTENTS, detect_intent
from brain.core.llm import get_llm
from brain.core.store import REPO_ROOT, Stores, now_iso
from brain.core.video import ingest_video, video_id_from_url

app = FastAPI(title="Trading Brain", version="0.1.0")
stores = Stores()

STATIC_DIR = Path(__file__).parent / "static"
PINE_DIR = REPO_ROOT / "pine"


def _strategy() -> dict | None:
    return stores.strategy.read()


def _save_strategy(s: dict) -> dict:
    s = generator.normalize_strategy(s, revision=s.get("revision", 1))
    stores.strategy.write(s)
    return s


def _sync_pine(s: dict) -> None:
    """Regenerate the indicator/strategy files whenever the brain changes."""
    PINE_DIR.mkdir(exist_ok=True)
    (PINE_DIR / "indicator.pine").write_text(generator.generate_indicator(s), encoding="utf-8")
    (PINE_DIR / "strategy.pine").write_text(generator.generate_strategy(s), encoding="utf-8")


def _ensure_seed() -> None:
    if stores.strategy.read() is None:
        _save_strategy(generator.default_strategy())
        _sync_pine(_strategy())


def _context_lines() -> str:
    s = _strategy()
    if not s:
        return "No strategy loaded yet."
    lines = [
        f"Strategy: {s['name']} (rev {s.get('revision', 1)})",
        f"Source: {s.get('source_video') or 'built-in'} · {s.get('timeframe') or 'any'}m · {s.get('market', 'any')}",
    ]
    lines += [l for l in generator.summarize_rules(s).splitlines()]
    stats = learn.review(stores, s)
    if stats["completed"]:
        lines.append(
            f"Signals seen: {stats['signals']} · completed {stats['completed']} · "
            f"win rate {stats['winRate'] or 0:.0%}"
        )
    return "\n".join(lines)


def _suggestions() -> list[dict]:
    stats = learn.review(stores, _strategy())
    return [stats["suggestion"]] if stats.get("suggestion") else []


def _chat_reply(user_message: str) -> str:
    settings = stores.settings.read()
    llm = get_llm(settings)
    return llm.reply(user_message, _context_lines())


# ────────────────────────────────────────────────────────────────────────────
# Static UI
# ────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ────────────────────────────────────────────────────────────────────────────
# Chat + learning endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty message")
    stores.add_message("user", text)

    # 1) a video link teaches the brain a new strategy
    vid = video_id_from_url(text)
    if vid:
        video, note = await ingest_video(text)
        stores.add_video(
            {k: video[k] for k in ("id", "title", "url", "transcript_source")}
        )
        extraction = await get_llm(stores.settings.read()).extract(
            video["transcript"], video
        )
        name = extraction.get("name")
        if not name or name == "Learned Strategy":
            name = f"Learned: {video['title'][:40]}" if video["title"] else "Learned Strategy"
        strategy = generator.normalize_strategy(
            extraction,
            name=name,
            source_video=video["url"],
        )
        strategy["transcript"] = video["transcript"][:8000]
        strategy = _save_strategy(strategy)
        _sync_pine(strategy)
        reply = (
            f"📺 Watched **“{video['title']}”** ({note}).\n\n"
            f"Here's what I learned:\n\n{generator.summarize_rules(strategy)}\n\n"
            f"✅ **{strategy['name']}** (rev {strategy['revision']}) is now your live strategy.\n"
            f"📄 `pine/indicator.pine` — indicator with BUY/SELL signals + webhook alerts to me\n"
            f"📄 `pine/strategy.pine` — backtest version of the same rules\n\n"
            f"Paste the indicator into TradingView, then create two alerts on the "
            f"“Brain BUY/SELL” alert conditions pointing at `/webhook/tv` so I can start learning. 🧠"
        )
        if strategy.get("notes"):
            reply += "\n\n⚠️ Notes: " + " · ".join(strategy["notes"])
        stores.add_message("assistant", reply)
        return {"reply": reply, "strategy": strategy, "video": video["title"]}

    # 2) generic chat
    intent = detect_intent(text)
    reply = await _chat_reply(text)
    if intent == "status":
        reply = f"{INTENTS['status'][0]} — here's the current brain state:\n\n{_context_lines()}"
    if intent == "clear":
        stores.strategy.write(generator.default_strategy())
        stores.signals.write([])
        stores.messages.write([])
        _sync_pine(_strategy())
        reply = "🧹 Brain wiped — back to the built-in starter strategy. `pine/` regenerated."
    if intent == "learn":
        stats = learn.review(stores, _strategy())
        suggestions = _suggestions()
        reply = (
            f"🧠 Learning report:\n\n"
            f"• Signals received: **{stats['signals']}** ({stats['buys']} buy, {stats['sells']} sell)\n"
            f"• Completed trades: **{stats['completed']}**"
        )
        if stats["winRate"] is not None:
            reply += f" · win rate **{stats['winRate']:.0%}**"
        reply += "\n"
        if suggestions:
            s = suggestions[0]
            reply += (
                f"\n💡 Suggestion: **{s['explain']}**\n"
                f"_({s['reason']})_\n\n"
                f"Say **“apply the suggestion”** and I'll update the strategy + regenerate the indicator."
            )
        else:
            reply += "\n\nNothing to tune yet — keep those TradingView alerts flowing. "
            reply += "Once I've seen 5 completed BUY→SELL pairs I can start proposing improvements."
    stores.add_message("assistant", reply)
    return {"reply": reply, "strategy": _strategy()}


@app.post("/api/apply-suggestion")
async def apply_suggestion():
    suggestions = _suggestions()
    if not suggestions:
        raise HTTPException(404, "no suggestion available")
    patch = suggestions[0]["patch"]
    strategy = generator.apply_patch(_strategy(), patch)
    _save_strategy(strategy)
    _sync_pine(strategy)
    stores.add_message(
        "assistant",
        f"✅ Applied learning suggestion (rev {strategy['revision']}). `pine/` regenerated. "
        f"Re-copy `pine/indicator.pine` into TradingView.",
    )
    return {"ok": True, "strategy": strategy}


# ────────────────────────────────────────────────────────────────────────────
# State + settings + pine output
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/state")
async def state():
    strategy = _strategy()
    stats = learn.review(stores, strategy)
    return {
        "strategy": strategy,
        "stats": stats,
        "suggestions": _suggestions(),
        "provider": get_llm(stores.settings.read()).label,
        "videos": stores.videos.read()[-10:],
        "messages": stores.messages.read()[-60:],
    }


@app.get("/api/pine/{kind}")
async def pine(kind: str):
    if kind not in ("indicator", "strategy"):
        raise HTTPException(404, "use /api/pine/indicator or /api/pine/strategy")
    path = PINE_DIR / f"{kind}.pine"
    if not path.exists():
        _sync_pine(_strategy())
    return JSONResponse(
        {"code": path.read_text(encoding="utf-8"), "kind": kind}
    )


@app.post("/api/settings")
async def settings(req: Request):
    body = await req.json()
    cur = stores.settings.read()
    for key in ("provider", "api_key", "model", "native_video"):
        if key in body and body[key] is not None:
            cur[key] = body[key]
    stores.settings.write(cur)
    return {"provider": get_llm(cur).label}


# ────────────────────────────────────────────────────────────────────────────
# TradingView alert webhook — the live learning loop
# ────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/tv")
async def webhook(req: Request):
    payload = await req.json()
    if payload.get("strategy") == "__ping__":
        return {"ok": True, "brain": "alive"}
    payload["received_at"] = now_iso()
    learn.record(stores, payload)
    stats = learn.review(stores, _strategy())
    return {"ok": True, "completed": stats["completed"]}


# ────────────────────────────────────────────────────────────────────────────
# Offline/demo helpers
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/demo-video")
async def demo_video():
    """Feed the brain the built-in demo transcript (offline test of the pipeline)."""
    from brain.core.video import DEMO_NOTE, DEMO_TRANSCRIPT

    video = {"id": "demo", "title": "Demo: 15m EMA + RSI Scalping Strategy", "url": None}
    extraction = await get_llm(stores.settings.read()).extract(DEMO_TRANSCRIPT, video)
    strategy = generator.normalize_strategy(
        extraction, name="EMA + RSI Scalp", source_video="demo://ema-rsi-scalp"
    )
    strategy["transcript"] = DEMO_TRANSCRIPT
    _save_strategy(strategy)
    _sync_pine(strategy)
    stores.add_message(
        "assistant",
        f"📺 Watched the demo video ({DEMO_NOTE}).\n\n{generator.summarize_rules(strategy)}"
    )
    return {"reply": generator.summarize_rules(strategy), "strategy": strategy}


@app.post("/api/webhook/simulate")
async def simulate():
    """Simulate a completed BUY→SELL pair (for testing the learning loop offline)."""
    strategy = _strategy()
    buys = [s for s in stores.signals.read() if s.get("action") == "BUY"]
    sells = [s for s in stores.signals.read() if s.get("action") == "SELL"]
    price = 100.0
    buys_n, sells_n = len(buys), len(sells)
    recorded = []
    if buys_n <= sells_n:
        learn.record(stores, {
            "strategy": strategy["name"], "revision": strategy.get("revision", 1),
            "action": "BUY", "symbol": "DEMO", "interval": "15",
            "price": price, "time": now_iso(),
        })
        buys_n += 1
        recorded.append("BUY")
    if sells_n < buys_n:
        learn.record(stores, {
            "strategy": strategy["name"], "revision": strategy.get("revision", 1),
            "action": "SELL", "symbol": "DEMO", "interval": "15",
            "price": 101.0, "time": now_iso(),
        })
        sells_n += 1
        recorded.append("SELL")
    return {"ok": True, "recorded": recorded, "stats": learn.review(stores, strategy)}


# ────────────────────────────────────────────────────────────────────────────
# Boot: make sure a strategy + pine files always exist
# ────────────────────────────────────────────────────────────────────────────

_ensure_seed()
