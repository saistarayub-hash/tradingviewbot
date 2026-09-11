# 🧠 Trading Brain — an indicator that learns from videos

A **TradingView indicator** with a **brain**: paste a YouTube video link into the
chat, the brain watches it (captions), extracts the trading strategy, and
generates a Pine Script v6 indicator for TradingView. The indicator fires
**alert webhooks back to the brain**, which logs every BUY/SELL, measures the
results, and proposes improvements — that's the learning loop.

```
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  YOU                          │         │  TRADING BRAIN (this repo)   │
│  • paste video link in chat   │ ──────▶ │  • fetch YouTube captions    │
│  • paste indicator into TV    │         │  • extract strategy (LLM or  │
│  • create 2 alerts → webhook  │         │    offline rule engine)      │
└──────────────────────────────┘         │  • generate pine/*.pine      │
                                         └──────────────┬───────────────┘
                                                        │
┌──────────────────────────────┐                       │
│  TRADINGVIEW                 │  alert webhooks        │
│  pine/indicator.pine         │ ──── BUY/SELL ───────▶ │
│  (signals + stop/target)     │         └─▶ learn: win rate, tune risk, suggest
└──────────────────────────────┘
```

## Quick start

```bash
cd tradingviewbot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn brain.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** → chat UI. Works immediately with **zero API
keys** (offline rule engine + built-in demo video). Click **🎥 Demo video** to
see the whole pipeline: transcript → strategy → `pine/indicator.pine`.

### Feed it a real video

Paste a YouTube link with subtitles into the chat, e.g. a strategy tutorial.
The brain fetches the caption track, extracts the rules, and regenerates:

| File | What it is |
|---|---|
| `pine/indicator.pine` | The **indicator**: plots BUY/SELL arrows, tracks the stop & target levels, fires `alertcondition`s (BUY/SELL) you can attach webhooks to |
| `pine/strategy.pine` | The **same rules** as a TradingView `strategy()` script → use the Strategy Tester to backtest |
| `data/strategy.json` | The canonical strategy the brain "knows" (rules, stop, target, revision) |

### Connect TradingView (the learning loop)

1. TradingView → Pine Editor → paste `pine/indicator.pine` → **Add to chart**.
2. Make your server reachable from the internet (e.g. `cloudflared tunnel --url http://localhost:8000`).
3. Create two alerts on the chart, both *webhook* type, URL `https://YOUR-TUNNEL/webhook/tv`
   (the exact URL is shown in the app):
   - Condition **“Brain BUY — …”** → *Once per bar close*
   - Condition **“Brain SELL — …”** → *Once per bar close*
4. Let it trade. The brain pairs BUY→SELL per symbol/timeframe, computes win
   rate, and after 5 completed trades starts proposing tweaks (e.g. widen the
   ATR stop if it keeps getting stopped out). You approve — the brain never
   silently changes live logic. Click **Simulate pair** to test the loop offline.

## Teaching the brain (rules it understands)

The brain maps what it hears onto a fixed, safe set of rules that it knows how
to compile into Pine — it never free-styles code, so the generated indicator
always compiles. Rule ids:

- **Entry:** `ema_cross_up`, `sma_cross_up`, `macd_cross_up`, `rsi_cross_up`,
  `bb_lower_touch`, `support_bounce`, `breakout_high`, `supertrend_up`
- **Exit:** `ema_cross_down`, `sma_cross_down`, `macd_cross_down`,
  `rsi_cross_down`, `bb_upper_touch`, `supertrend_down`, `resistance_reject`
- **Filters:** `rsi_below`, `rsi_above`, `price_above_ema`, `price_below_ema`,
  `vwap_above`, `vwap_below`, `adx_above`, `volume_spike`, `session`
- **Stop:** ATR-multiple / percent / swing low · **Target:** R:R / percent / opposite

The rule engine (offline) understands plain-English patterns like *“9 EMA
crosses above the 21 EMA”*, *“RSI below 50”*, *“2 ATR stop”*, *“2 to 1 risk
reward”*. With an LLM key it extracts far more.

## Optional: real AI provider

Settings (⚙️ in the app) → pick a provider + key (stored only in
`data/settings.json`, never leaves your machine):

- **Google Gemini** (recommended, free tier): can watch YouTube videos
  natively (no transcription needed). Key from <https://aistudio.google.com>.
- **OpenAI**: used on the transcript text.
- **Offline**: no key, rule-engine extraction, everything still works.

No captions on a video? Install `yt-dlp` + `faster-whisper` for audio
transcription (see `requirements.txt` extras).

## Project layout

```
brain/
  main.py            FastAPI app: chat, webhook, settings, state
  core/
    store.py         JSON persistence (data/ dir)
    video.py         YouTube → captions → transcript
    insight.py       offline rule engine (intent + strategy extraction)
    llm.py           providers: mock / gemini / openai
    generator.py     strategy JSON → Pine Script v6 (deterministic)
    learn.py         BUY/SELL pairing, win rate, tuning suggestions
  static/index.html  the chat + indicator UI
pine/
  indicator.pine     the indicator (regenerated whenever the brain learns)
  strategy.pine      backtest version
data/                runtime state (git-ignored): strategy, signals, videos, chat
```

## Disclaimer

Educational tool. The brain summarises videos, generates *candidate* strategy
code, and tunes numbers — it is **not financial advice**, it does not guarantee
the extracted rules match what the video author meant, and generated signals
should be backtested and paper-traded before any real use. Trading involves
substantial risk.
