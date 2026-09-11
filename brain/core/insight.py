"""Rule-based strategy extraction (offline brain) + intent detection.

This is what makes the brain work with **no API key**: scan a transcript for
known strategy phrases (EMA crossovers, RSI thresholds, ATR stops, risk:reward,
timeframes...) and map them onto the canonical strategy schema.

It is deliberately conservative — it only emits rule ids the Pine generator
understands, and it always records uncertainties in ``notes``.
"""
from __future__ import annotations

import re

from .generator import MIRROR

STOP_PATTERNS = {
    "percent": [
        r"stop(?:-?loss)?\s+(?:of|at)?\s*([0-9.]+)\s*%",
        r"([0-9.]+)\s*%\s+stop",
    ],
    "atr": [
        r"stop\s*(?:of|at)?\s*([0-9.]+)\s*(?:x|times)?\s*atr",
        r"stop(?:-?loss)?\s+(?:of|at)?\s*([0-9.]+)\s*(?:x|times)?\s*atr",
        r"([0-9.]+)\s*atr\s+(?:stop|below)",
    ],
    "swing": [
        r"stop\s+(?:below|under)\s+(?:the\s+)?(?:recent\s+|previous\s+)?(?:swing\s+)?low",
        r"swing\s+low\s+stop",
    ],
}

INTENTS = {
    "apply": ("Applying 💪",),
    "help": ("Can do 👇",),
    "status": ("Brain status 📊",),
    "learn": ("Learning mode 🧠",),
    "clear": ("Fresh brain 🧹",),
    "save": ("Saving 💾",),
    "greet": ("Hey! 👋",),
    "thanks": ("🙏",),
    "chat": ("Chatting 💬",),
}

# ── number-word normalization ───────────────────────────────────────────────

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUM_WORD = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[-\s]+(" + "|".join(_ONES) + r"))?|("
    + "|".join(_ONES) + r"))\b"
)


def _num_word_repl(m: re.Match) -> str:
    tens, ones, single = m.group(1), m.group(2), m.group(3)
    if tens:
        value = _TENS[tens] + (_ONES[ones] if ones else 0)
    else:
        value = _ONES[single]
    return str(value)


def normalize_numbers(text: str) -> str:
    """'nine EMA and the twenty one EMA' → '9 EMA and the 21 EMA'."""
    return _NUM_WORD.sub(_num_word_repl, (text or "").lower())


def detect_intent(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(hi|hello|hey|howdy|yo)\b", t):
        return "greet"
    if re.search(r"\b(thanks|thank you|thx)\b", t):
        return "thanks"
    if re.search(r"\bapply\b", t):
        return "apply"
    if re.search(r"\b(help|what can you do|how does this work|how do i|commands)\b", t):
        return "help"
    if re.search(r"\b(status|current strategy|what.?s your strategy|show strategy|brain state)\b", t):
        return "status"
    if re.search(r"\b(learn|improve|optimize|review|performance|win rate)\b", t):
        return "learn"
    if re.search(r"\b(clear|reset|forget|wipe)\b", t):
        return "clear"
    if re.search(r"\b(save|keep|remember)\b", t):
        return "save"
    return "chat"


STRATEGY_HINT = re.compile(
    r"\b(ema|sma|vwap|macd|rsi|bollinger|atr|ichimoku|supertrend|adx|crossover|crosses?"
    r"|breakout|breakdown|stop[- ]?loss|take[- ]?profit|risk[- ]?reward|entry|exit|"
    r"scalp|strategy|setup|indicator|signal|moving average|long\s+when|short\s+when)\b",
    re.IGNORECASE,
)


def is_strategy_text(text: str) -> bool:
    """True when a chat message looks like the user is describing a strategy
    directly (no video link needed)."""
    t = (text or "").strip()
    return len(t) >= 24 and bool(STRATEGY_HINT.search(t))


SHORT_CTX_RE = re.compile(
    r"\b(short signal|short entry|go\s+short|shorting|short\s+(?:when|if|on|at|the)"
    r"|open(?:ing)?\s+a\s+short|short\s+position|short\s+trade)\b"
)


def _nums(pattern: str, text: str):
    return [float(x) for x in re.findall(pattern, text)]


def _dedupe(rules: list[dict]) -> list[dict]:
    """Keep the first occurrence of each rule id."""
    seen: set[str] = set()
    out = []
    for r in rules:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def _first_num(pattern: str, text: str, default):
    nums = _nums(pattern, text)
    return nums[0] if nums else default


def extract_heuristic(text: str) -> dict:
    t = normalize_numbers(text or "")
    entry, exits, filters = [], [], []
    short_extra: list[dict] = []
    short_filters_direct: list[dict] = []
    notes: list[str] = []

    # ── Moving-average crossovers ──────────────────────────────────────
    seen_pairs: set[tuple[int, int]] = set()
    for m in re.finditer(
        r"\b(\d+)\s*(?:period\s*)?ema\b(.{0,30}?)\b(\d+)\s*(?:period\s*)?ema\b", t
    ):
        if not re.search(r"\b(and|cross\w*)\b", m.group(2)):
            continue  # not a crossover pairing (e.g. two unrelated EMAs)
        pair = (int(m.group(1)), int(m.group(3)))
        fast, slow = sorted(pair)
        pair = (fast, slow)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        entry.append({"id": "ema_cross_up", "params": {"fast": fast, "slow": slow}})
        exits.append({"id": "ema_cross_down", "params": {"fast": fast, "slow": slow}})
    # EMA stack chain ("10 EMA above the 20 above the 50") — computed first
    # so the single-EMA filter below can ignore those spans
    chain = re.search(
        r"\b(\d+)\s*(?:period\s*)?ema\b(?:\s+is)?\s+above\s+(?:the\s+)?"
        r"(\d+)\s*(?:period\s*)?ema\b(?:\s+is)?\s+above\s+(?:the\s+)?"
        r"(\d+)\s*(?:period\s*)?ema\b", t
    )
    stack_kw = bool(re.search(r"\bema\s*stack|stacked\s*emas|stacked\s+moving\s+averages\b", t))

    # single-EMA mentions: "price above the 200 ema" etc. (NOT "crosses above")
    for m in re.finditer(r"\b(above|below)\s+the\s+(\d+)\s*(?:period\s*)?ema\b", t):
        if re.search(r"\bcross(?:es|ing)?(?:\s+back)?\s*$", t[max(0, m.start() - 16):m.start()]):
            continue
        if chain and chain.start() <= m.start() <= chain.end():
            continue
        filters.append({"id": f"price_{m.group(1)}_ema", "params": {"length": int(m.group(2))}})

    # generic "ema crossover" without numbers → 9/21 default
    if re.search(r"\bema\s+cross(?:es|ing|over|s)?\b", t) and not any(
        r["id"] == "ema_cross_up" for r in entry
    ):
        entry.append({"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}})
        exits.append({"id": "ema_cross_down", "params": {"fast": 9, "slow": 21}})

    # ── RSI ────────────────────────────────────────────────────────────
    rsi_len = int(_first_num(r"\brsi\s*(?:\(|with)?\s*(?:a\s+)?(?:length\s+of\s+)?(\d+)", t, 14))
    if re.search(r"\brsi\b", t):
        below_m = re.search(r"rsi\s+(?:(?:is|to\s+be)\s+)?(?:below|under|less than|beneath)\s*(\d+)", t)
        above_m = re.search(r"rsi\s+(?:(?:is|to\s+be)\s+)?(?:above|over|more than|greater than)\s*(\d+)", t)
        short_cut_m = SHORT_CTX_RE.search(t)
        short_cut = short_cut_m.start() if short_cut_m else len(t)
        oversold = re.search(r"rsi\s+(?:(?:is|to\s+be)\s+)?(?:below\s+)?(?:30|oversold)", t)
        overbought = re.search(r"rsi\s+(?:(?:is|to\s+be)\s+)?(?:above\s+)?(?:70|overbought)", t)
        cross_up = re.search(
            r"rsi\s+cross(?:es|ing|ed)?\s+(?:up|above|back above)\s*(?:through|above)?\s*(\d+)", t
        )
        cross_dn = re.search(
            r"rsi\s+cross(?:es|ing|ed)?\s+(?:down|below|back below)\s*(?:through|below)?\s*(\d+)", t
        )
        if below_m:
            rule = {"id": "rsi_below", "params": {"length": rsi_len, "level": int(below_m.group(1))}}
            (short_filters_direct if short_cut_m and below_m.start() >= short_cut else filters).append(rule)
        if above_m:
            rule = {"id": "rsi_above", "params": {"length": rsi_len, "level": int(above_m.group(1))}}
            (short_filters_direct if short_cut_m and above_m.start() >= short_cut else filters).append(rule)
        if oversold and not cross_up:
            entry.append({"id": "rsi_cross_up", "params": {"length": rsi_len, "level": 30}})
            notes.append("Interpreted 'oversold RSI' as RSI crossing back above 30.")
        if overbought and not cross_dn:
            exits.append({"id": "rsi_cross_down", "params": {"length": rsi_len, "level": 70}})
            notes.append("Interpreted 'overbought RSI' as RSI crossing back below 70.")
        if cross_up:
            entry.append({"id": "rsi_cross_up", "params": {"length": rsi_len, "level": int(cross_up.group(1))}})
        if cross_dn:
            exits.append({"id": "rsi_cross_down", "params": {"length": rsi_len, "level": int(cross_dn.group(1))}})

    # ── Ichimoku ────────────────────────────────────────────────────────
    if re.search(r"\bichimoku\b", t):
        above_m = re.search(r"above\s+the\s+(?:ichimoku\s+)?cloud", t)
        below_m = re.search(r"below\s+the\s+(?:ichimoku\s+)?cloud", t)
        if above_m:
            filters.append({"id": "ichimoku_above_cloud", "params": {}})
            if not entry:
                entry.append({"id": "ichimoku_cloud_up", "params": {}})
        if below_m:
            window = t[max(0, below_m.start() - 25):below_m.start()]
            if re.search(r"\b(exit|sell|close(?:s|d)?|cover)\b", window):
                exits.append({"id": "ichimoku_cloud_down", "params": {}})
            else:
                filters.append({"id": "ichimoku_below_cloud", "params": {}})
                short_extra.append({"id": "ichimoku_cloud_down", "params": {}})
        if not above_m and not below_m:
            entry.append({"id": "ichimoku_cloud_up", "params": {}})
            exits.append({"id": "ichimoku_cloud_down", "params": {}})
            notes.append("Ichimoku mentioned — used price crossing the cloud up/down as entry/exit.")
        if not exits:
            exits.append({"id": "ichimoku_cloud_down", "params": {}})

    # ── EMA stack filter ───────────────────────────────────────────────
    if chain or stack_kw:
        if chain:
            fast, mid, slow = int(chain.group(1)), int(chain.group(2)), int(chain.group(3))
        else:
            fast, mid, slow = 10, 20, 50
        filters.append({"id": "ema_stack_bull", "params": {"fast": fast, "mid": mid, "slow": slow}})

    # ── HalfTrend ───────────────────────────────────────────────────────
    halftrend_seen = bool(re.search(r"\bhalf\s*tren(?:d)?\b", t))
    if halftrend_seen:
        entry.append({"id": "halftrend_up", "params": {}})
        exits.append({"id": "halftrend_down", "params": {}})

    # ── MACD ───────────────────────────────────────────────────────────
    if re.search(r"\bmacd\b", t):
        if re.search(r"macd\s+cross\w*\s*(up|above)|bullish\s+macd|cross\w*\s+(?:up\s+|above\s+)?(?:on\s+)?\w*macd", t):
            entry.append({"id": "macd_cross_up", "params": {}})
        if re.search(r"macd\s+cross\w*\s*(down|below)|bearish\s+macd|cross\w*\s+(?:down\s+|below\s+)?\w*macd", t):
            exits.append({"id": "macd_cross_down", "params": {}})
        if not any(r["id"].startswith("macd") for r in entry + exits):
            entry.append({"id": "macd_cross_up", "params": {}})
            exits.append({"id": "macd_cross_down", "params": {}})
            notes.append("MACD mentioned without direction — used MACD cross up/down as entry/exit.")

    # ── Bollinger Bands ────────────────────────────────────────────────
    if re.search(r"\b(bollinger|bb|bands)\b", t):
        if re.search(r"(lower|bottom)\s*(band|bollinger)", t):
            entry.append({"id": "bb_lower_touch", "params": {}})
        if re.search(r"(upper|top)\s*(band|bollinger)", t):
            exits.append({"id": "bb_upper_touch", "params": {}})
        if not any(r["id"].startswith("bb_") for r in entry + exits):
            entry.append({"id": "bb_lower_touch", "params": {}})
            exits.append({"id": "bb_upper_touch", "params": {}})

    # ── VWAP / ADX / volume / session ─────────────────────────────────
    if re.search(r"\bvwap\b", t):
        if re.search(r"(above|over)\s+(the\s+)?vwap", t):
            filters.append({"id": "vwap_above", "params": {}})
        elif re.search(r"(below|under)\s+(the\s+)?vwap", t):
            filters.append({"id": "vwap_below", "params": {}})
        else:
            filters.append({"id": "vwap_above", "params": {}})
            notes.append("VWAP mentioned — assumed price above VWAP as a filter.")
    if re.search(r"\badx\b", t):
        level = int(_first_num(r"adx\s*(?:above|over|greater than|>|is\s+at least)?\s*(\d+)", t, 25))
        filters.append({"id": "adx_above", "params": {"length": 14, "level": level}})
    if re.search(r"\b(volume spike|high volume|volume (?:surge|explosion|increase))\b", t):
        filters.append({"id": "volume_spike", "params": {}})

    # ── Breakout / support / breakdown ─────────────────────────────────
    if re.search(r"\b(breakout|breaks? (?:out )?(?:above|through))\b", t):
        lb = int(_first_num(r"(\d+)\s*(?:bar|day|hour|candle)", t, 20))
        entry.append({"id": "breakout_high", "params": {"lookback": lb}})
    for m in re.finditer(r"\bbreakdown\b|\bbreaks?\s+(?:down|below)\b", t):
        lb = int(_first_num(r"(\d+)\s*(?:bar|day|hour|candle)", t, 20))
        window = t[max(0, m.start() - 25):m.start()]
        rule = {"id": "breakdown_low", "params": {"lookback": lb}}
        if re.search(r"\b(exit|sell|close(?:s|d)?|cover)\b", window):
            exits.append(rule)              # "exit when price breaks below X" → long exit
        else:
            short_extra.append(rule)        # standalone "breaks below" → short entry
    if re.search(r"\b(support bounce|bounces? off (?:the )?support|support level)\b", t):
        entry.append({"id": "support_bounce", "params": {}})

    # ── Short-side detection ───────────────────────────────────────────
    short_hint = bool(SHORT_CTX_RE.search(t)) or bool(short_extra)

    # ── Stop / target ──────────────────────────────────────────────────
    stop = None
    for stype, pats in STOP_PATTERNS.items():
        for pat in pats:
            if re.search(pat, t):
                num = _first_num(pat, t, None)
                if stype == "percent" and num:
                    stop = {"type": "percent", "params": {"pct": num}}
                elif stype == "atr" and num:
                    stop = {"type": "atr", "params": {"length": 14, "mult": num}}
                elif stype == "swing":
                    stop = {"type": "swing", "params": {"left": 10, "right": 10}}
                break
        if stop:
            break

    target = None
    # "two to one risk reward", "2:1 R:R", "risk reward of 1:2"
    rr_before = re.search(
        r"([0-9.]+)\s*(?:to|:)\s*([0-9.]+)\s*(?:risk[\s-]?reward|risk to reward|r\s*:\s*r)", t
    )
    rr_after = re.search(
        r"(?:risk[\s-]?reward|risk to reward|r\s*:\s*r)\s*(?:of\s*)?(?:is\s*)?(?:at least\s*)?"
        r"([0-9.]+)\s*(?:to|:)\s*([0-9.]+)?", t
    )
    if rr_before:
        a, b = float(rr_before.group(1)), float(rr_before.group(2))
        target = {"type": "rr", "params": {"rr": round(a / b, 2) if b else a}}
    elif rr_after:
        a, b = float(rr_after.group(1)), float(rr_after.group(2) or 1)
        target = {"type": "rr", "params": {"rr": round(a / b, 2)}}
    elif re.search(r"\btarget\s*(?:of|at|is)?\s*([0-9.]+)\s*r\b|\brr\s*(?:of|at|is)?\s*([0-9.]+)\b", t):
        m = re.search(r"\btarget\s*(?:of|at|is)?\s*([0-9.]+)\s*r\b|\brr\s*(?:of|at|is)?\s*([0-9.]+)\b", t)
        rr_val = float(m.group(1) or m.group(2) or 2.0)
        target = {"type": "rr", "params": {"rr": rr_val}}
    elif re.search(r"\btake profit\b|\btp\b|\bprofit target\b", t):
        pct = _first_num(r"(?:take[- ]?profit|profit target|tp)\s*(?:of|at)?\s*([0-9.]+)\s*%", t, 2.0)
        target = {"type": "percent", "params": {"pct": pct}}
    elif stop and not re.search(r"\btake profit\b|\btarget\b", t):
        target = {"type": "opposite", "params": {}}

    if halftrend_seen:
        # default risk from BigBeluga's engine: SL = baseRisk(3) × ATR(100)/2
        if stop is None:
            stop = {"type": "atr", "params": {"length": 100, "mult": 1.5}}
        if target is None:
            target = {"type": "rr", "params": {"rr": 3.0}}
        notes.append("HalfTrend risk: SL = 1.5×ATR(100) below entry, target 3R (TP3).")

    if not entry:
        notes.append("Couldn't find a clear entry signal — defaulted to EMA 9/21 crossover.")
    if not exits:
        notes.append("No exit signal found — position closes on stop/target (or max bars).")

    # ── Mirror the long side when the video teaches shorting ──────────
    side = "both" if short_hint else "long"
    short_entry: list[dict] = []
    short_exit: list[dict] = []
    short_filters: list[dict] = []
    if short_hint:
        short_entry = [
            {"id": MIRROR.get(r["id"], r["id"]), "params": dict(r.get("params") or {})}
            for r in entry
        ]
        short_exit = [
            {"id": MIRROR.get(r["id"], r["id"]), "params": dict(r.get("params") or {})}
            for r in exits
        ]
        direct_ids = {r["id"] for r in short_filters_direct}
        short_filters = list(short_filters_direct) + [
            {"id": MIRROR.get(r["id"], r["id"]), "params": dict(r.get("params") or {})}
            for r in filters
            if MIRROR.get(r["id"], r["id"]) not in direct_ids
        ]
        notes.append("Short-side rules were mirrored from the long-side rules.")
    short_entry += short_extra
    short_entry = _dedupe(short_entry)
    short_exit = _dedupe(short_exit)
    short_filters = _dedupe(short_filters)
    entry = _dedupe(entry)
    exits = _dedupe(exits)
    filters = _dedupe(filters)

    # ── Timeframe / market ─────────────────────────────────────────────
    tf = None
    tf_map = {
        r"\b1\s*min(?:ute)?s?\b": "1",
        r"\b5\s*min(?:ute)?s?\b": "5",
        r"\b15\s*min(?:ute)?s?\b": "15",
        r"\b30\s*min(?:ute)?s?\b": "30",
        r"\b1\s*hour\b": "60",
        r"\b4\s*hour\b": "240",
        r"\bdaily\b|\b1d\b": "1D",
    }
    for phrase, val in tf_map.items():
        if re.search(phrase, t):
            tf = val
            break

    market = "any"
    if re.search(r"\b(crypto|bitcoin|btc|eth|ethereum)\b", t):
        market = "crypto"
    elif re.search(r"\b(forex|eurusd|gbpusd|pip)\b", t):
        market = "forex"
    elif re.search(r"\b(stocks?|spy|nasdaq)\b", t):
        market = "stocks"

    return {
        "name": "Learned Strategy",
        "timeframe": tf,
        "market": market,
        "side": side,
        "summary": "Extracted by the brain's rule engine from the video transcript.",
        "notes": notes[:6],
        "rules": {
            "entry": entry or [{"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}}],
            "exit": exits,
            "short_entry": short_entry,
            "short_exit": short_exit,
            "filters": filters,
            "short_filters": short_filters,
        },
        "stop": stop,
        "target": target,
    }
