# 🧠 Trading Brain — an indicator that learns from videos

A **TradingView indicator** with a **brain**: type your strategy into the chat
in plain English, and the brain compiles it into a neat Pine Script v6
indicator where every BUY/SELL/SHORT/COVER signal shows a **label with the
reasons it fired**, plus a live position panel on the chart. You can also
paste a **YouTube video link** and the brain learns the strategy from the
captions. Alert webhooks feed signals back to the brain, which tracks
performance and proposes improvements — the learning loop.

```
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  YOU                          │         │  TRADING BRAIN (this repo)   │
│  • type strategy in chat      │ ──────▶ │  • parse rules (engine/LLM) │
│    "buy when EMA 9 crosses    │         │  • compile pine/*.pine      │
│     above EMA 21, RSI < 50,   │         │  • every signal labelled    │
│     stop 2 ATR, target 2R"    │         │    with its reasons         │
│  • (or paste a YouTube link)  │         └──────────────┬───────────────┘
└──────────────────────────────┘                        │
                                                        │
┌──────────────────────────────┐                       │
│  TRADINGVIEW                 │  alert webhooks        │
│  pine/indicator.pine         │ ── BUY/SELL/SHORT/COVER│
│  signals + reasons + panel   │         └─▶ learn: win rate, suggest, tune
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
keys** (offline rule engine). The quickest way to use it: **type your
strategy** in the chat, e.g.

> buy when the 9 EMA crosses above the 21 EMA and RSI is below 50, exit when
> it crosses back below. Stop 2 ATR, target 2R. 15 minute chart. Also short
> when it crosses below with RSI above 50.

…and the brain rebuilds the indicator instantly. Quick-example chips sit under
the chat box (EMA cross, Ichimoku + EMA stack, breakout scalp).

### Feed it a video instead

Paste a YouTube link with subtitles into the chat (e.g. a strategy tutorial).
The brain fetches the caption track, extracts the rules, and regenerates the
scripts — same output as typing the strategy.

| File | What it is |
|---|---|
| `pine/indicator.pine` | The **indicator**, organised in neat blocks: **LONG BLOCK** / **SHORT BLOCK** conditions + engines, on-chart **labels on every signal showing the reasons** it fired, a live **position panel** (top-right table), stop/target lines, and 4 `alertcondition`s |
| `pine/strategy.pine` | The **same rules** as a TradingView `strategy()` script (long + short) → use the Strategy Tester to backtest |
| `pine/halftrend_example.pine` | Reference copy of BigBeluga's original HalfTrend engine (CC BY-NC-SA 4.0) — the brain's `halftrend_*` rules implement the same logic with signals, reason labels and webhook alerts |
| `data/strategy.json` | The canonical strategy the brain "knows" (rules, side, stop, target, revision) |

The chat app also includes a **Signal dashboard** (recent webhook feed +
completed trades with win/loss) and a **learning panel** with live stats and
the brain's current suggestion.

### Connect TradingView (the learning loop)

1. TradingView → Pine Editor → paste `pine/indicator.pine` → **Add to chart**.
2. Make your server reachable from the internet (e.g. `cloudflared tunnel --url http://localhost:8000`).
3. Create up to four alerts on the chart, all *webhook* type, URL `https://YOUR-TUNNEL/webhook/tv`
   (the exact URL is shown in the app):
   - **“Brain BUY — …”** and **“Brain SELL — …”** (long side), *Once per bar close*
   - **“Brain SHORT — …”** and **“Brain COVER — …”** (short side, when your strategy is both-sides), *Once per bar close*
4. Let it trade. The brain pairs **BUY→SELL** (long) and **SHORT→COVER**
   (short) per symbol/timeframe, computes win rate per side, and after 5
   completed trades starts proposing tweaks (e.g. widen the ATR stop if it
   keeps getting stopped out). You approve — the brain never silently changes
   live logic. Click **Simulate 2 pairs** to test the loop offline and watch
   the **Signal dashboard** fill up.

## Teaching the brain (rules it understands)

The brain maps what it hears onto a fixed, safe set of rules that it knows how
to compile into Pine — it never free-styles code, so the generated indicator
always compiles. Strategies can be **long-only, short-only, or both sides**
(`side`); short rules live in `short_entry` / `short_exit` / `short_filters`
and mirror the long-side vocabulary. If a video teaches shorting, the brain
mirrors the long rules into the short buckets automatically.

Rule ids:

- **Signals:** `ema_cross_up/down`, `sma_cross_up/down`, `macd_cross_up/down`,
  `rsi_cross_up/down`, `bb_lower_touch`, `bb_upper_touch`, `supertrend_up/down`,
  `breakout_high`, `breakdown_low`, `support_bounce`, `resistance_reject`,
  `ichimoku_cloud_up/down`, `halftrend_up/down` (HalfTrend trend flips —
  amplitude + channel-deviation params, credited to everget / BigBeluga)
- **Filters:** `rsi_below`, `rsi_above`, `price_above_ema`, `price_below_ema`,
  `vwap_above`, `vwap_below`, `adx_above`, `volume_spike`, `session`,
  `ema_stack_bull/bear` (“20 EMA above the 50 above the 200”),
  `ichimoku_above/below_cloud`, `halftrend_bull/bear`
- **Stop:** ATR-multiple / percent / swing low (swing high for shorts) ·
  **Target:** R:R / percent / opposite

The rule engine (offline) understands plain-English patterns like *“9 EMA
crosses above the 21 EMA”*, *“RSI below 50”*, *“Ichimoku cloud”*, *“2 ATR
stop”*, *“2 to 1 risk reward”*, *“that's your short signal”*. With an LLM key
it extracts far more.

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
