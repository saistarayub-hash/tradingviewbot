"""Strategy → Pine Script (v6) generator.

The brain never free-styles Pine code: it extracts a *strategy* (a small
JSON object of known rules) and this module compiles it into deterministic,
syntactically-safe Pine Script v6. Two artefacts are produced:

  * indicator           — plots BUY/SELL/SHORT/COVER signals, tracks
                          stop/target levels per side, flips positions and
                          fires ``alertcondition`` webhooks at the brain
                          (the live loop the brain learns from).
  * backtest strategy   — the same rules as a ``strategy()`` script so the
                          logic can be backtested on TradingView.

The strategy is either long-only, short-only, or both sides (``side``).
Short-side rules live in ``rules.short_entry`` / ``rules.short_exit`` /
``rules.short_filters`` and mirror the long-side vocabulary.

Rule ids understood by the brain (tell the LLM / heuristics only these exist):

  signals (directional):
    ema_cross_up/down, sma_cross_up/down, macd_cross_up/down,
    rsi_cross_up/down, bb_lower_touch, bb_upper_touch, supertrend_up/down,
    breakout_high, breakdown_low, support_bounce, resistance_reject,
    ichimoku_cloud_up/down
  filters (ongoing conditions):
    rsi_below, rsi_above, price_above_ema, price_below_ema,
    vwap_above, vwap_below, adx_above, volume_spike, session,
    ema_stack_bull, ema_stack_bear, ichimoku_above_cloud, ichimoku_below_cloud

  stop    : {"type": "atr"|"percent"|"swing", "params": {...}}
  target  : {"type": "rr"|"percent"|"opposite", "params": {...}}
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Canonical strategy schema
# ---------------------------------------------------------------------------

CANONICAL = {
    "name": "EMA Crossover",
    "source_video": None,
    "timeframe": "15",
    "market": "crypto",
    "side": "long",             # long | short | both
    "summary": "",
    "rules": {
        "entry": [],            # long entry signals
        "exit": [],             # long exit signals
        "short_entry": [],      # short entry signals
        "short_exit": [],       # short exit signals
        "filters": [],          # long-side ongoing conditions
        "short_filters": [],    # short-side ongoing conditions
    },
    "stop": None,       # {"type": "atr"|"percent"|"swing", "params": {...}}
    "target": None,     # {"type": "rr"|"percent"|"opposite", "params": {...}}
    "notes": [],
    "revision": 1,
    "updated_at": None,
}

KNOWN_RULE_IDS = {
    # directional signals
    "ema_cross_up", "ema_cross_down",
    "sma_cross_up", "sma_cross_down",
    "macd_cross_up", "macd_cross_down",
    "rsi_cross_up", "rsi_cross_down",
    "bb_lower_touch", "bb_upper_touch",
    "supertrend_up", "supertrend_down",
    "breakout_high", "breakdown_low",
    "support_bounce", "resistance_reject",
    "ichimoku_cloud_up", "ichimoku_cloud_down",
    # filters
    "rsi_below", "rsi_above",
    "price_above_ema", "price_below_ema",
    "vwap_above", "vwap_below",
    "adx_above", "volume_spike", "session",
    "ema_stack_bull", "ema_stack_bear",
    "ichimoku_above_cloud", "ichimoku_below_cloud",
}

# Long ↔ short mirroring (for "the same strategy, flipped" extractions).
MIRROR = {
    "ema_cross_up": "ema_cross_down", "ema_cross_down": "ema_cross_up",
    "sma_cross_up": "sma_cross_down", "sma_cross_down": "sma_cross_up",
    "macd_cross_up": "macd_cross_down", "macd_cross_down": "macd_cross_up",
    "rsi_cross_up": "rsi_cross_down", "rsi_cross_down": "rsi_cross_up",
    "bb_lower_touch": "bb_upper_touch", "bb_upper_touch": "bb_lower_touch",
    "supertrend_up": "supertrend_down", "supertrend_down": "supertrend_up",
    "breakout_high": "breakdown_low", "breakdown_low": "breakout_high",
    "support_bounce": "resistance_reject", "resistance_reject": "support_bounce",
    "ichimoku_cloud_up": "ichimoku_cloud_down",
    "ichimoku_cloud_down": "ichimoku_cloud_up",
    "rsi_below": "rsi_above", "rsi_above": "rsi_below",
    "price_above_ema": "price_below_ema", "price_below_ema": "price_above_ema",
    "vwap_above": "vwap_below", "vwap_below": "vwap_above",
    "ema_stack_bull": "ema_stack_bear", "ema_stack_bear": "ema_stack_bull",
    "ichimoku_above_cloud": "ichimoku_below_cloud",
    "ichimoku_below_cloud": "ichimoku_above_cloud",
    "adx_above": "adx_above", "volume_spike": "volume_spike", "session": "session",
}


def mirror_rule(rule: dict) -> dict:
    return {"id": MIRROR.get(rule["id"], rule["id"]), "params": dict(rule.get("params") or {})}


DEFAULT_STOP = {"type": "atr", "params": {"length": 14, "mult": 2.0}}
DEFAULT_TARGET = {"type": "rr", "params": {"rr": 2.0}}


def default_strategy() -> dict:
    """Seed strategy shipped with the repo (EMA 9/21 + RSI filter, both sides)."""
    s = dict(CANONICAL)
    s.update(
        name="EMA Crossover",
        timeframe="15",
        market="crypto",
        side="both",
        summary=(
            "Long when the 9 EMA crosses above the 21 EMA while RSI(14) is below 50, "
            "exit when it crosses back below; mirrored shorts when it crosses below "
            "with RSI above 50. 2×ATR stop, 2R target."
        ),
        rules={
            "entry": [
                {"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}},
                {"id": "rsi_cross_up", "params": {"length": 14, "level": 30}},
            ],
            "exit": [
                {"id": "ema_cross_down", "params": {"fast": 9, "slow": 21}},
            ],
            "filters": [
                {"id": "rsi_below", "params": {"length": 14, "level": 50}},
            ],
            "short_entry": [
                {"id": "ema_cross_down", "params": {"fast": 9, "slow": 21}},
                {"id": "rsi_cross_down", "params": {"length": 14, "level": 70}},
            ],
            "short_exit": [
                {"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}},
            ],
            "short_filters": [
                {"id": "rsi_above", "params": {"length": 14, "level": 50}},
            ],
        },
        stop=dict(DEFAULT_STOP),
        target=dict(DEFAULT_TARGET),
        notes=["Built-in starter strategy — feed the brain a video to replace it."],
        revision=1,
        updated_at=None,
    )
    return s


def normalize_strategy(
    raw, name: str | None = None, source_video: str | None = None, revision: int = 1
) -> dict:
    """Coerce a raw extraction (LLM JSON or heuristics dict) into canonical form."""
    s = default_strategy() if raw is None else dict(CANONICAL)
    if isinstance(raw, dict):
        for key in ("name", "timeframe", "market", "side", "summary", "notes", "stop", "target"):
            if key in raw and raw[key] is not None:
                s[key] = raw[key]
        rules = raw.get("rules", {})
        if isinstance(rules, dict):
            for bucket in ("entry", "exit", "short_entry", "short_exit", "filters", "short_filters"):
                if bucket in rules and isinstance(rules[bucket], list):
                    s["rules"][bucket] = _clean_rules(rules[bucket])
        elif isinstance(rules, list):
            cleaned = _clean_rules(rules)
            for rule in cleaned:
                if rule["id"] in _FILTER_IDS:
                    s["rules"]["filters"].append(rule)
                elif rule["id"] in _EXIT_IDS:
                    s["rules"]["exit"].append(rule)
                else:
                    s["rules"]["entry"].append(rule)
    side = str(s.get("side", "long")).lower()
    if side not in ("long", "short", "both"):
        side = "long"
    s["side"] = side

    # If the extractor said "short" but only filled long buckets, mirror them.
    if side == "short" and not s["rules"]["short_entry"] and s["rules"]["entry"]:
        s["rules"]["short_entry"] = [mirror_rule(r) for r in s["rules"]["entry"]]
        s["rules"]["short_exit"] = [mirror_rule(r) for r in s["rules"]["exit"]]
        s["rules"]["short_filters"] = [mirror_rule(r) for r in s["rules"]["filters"]]
        s["rules"]["entry"] = []
        s["rules"]["exit"] = []
        s["rules"]["filters"] = []

    if side != "short" and not s["rules"]["entry"]:
        s["rules"]["entry"] = [{"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}}]
    if side == "short" and not s["rules"]["short_entry"]:
        s["rules"]["short_entry"] = [{"id": "ema_cross_down", "params": {"fast": 9, "slow": 21}}]
    if name:
        s["name"] = name
    if source_video:
        s["source_video"] = source_video
    s["revision"] = int(revision or 1)
    s["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(s["stop"], dict) and s["stop"].get("type") not in ("atr", "percent", "swing"):
        s["stop"] = DEFAULT_STOP
    if isinstance(s["target"], dict) and s["target"].get("type") not in ("rr", "percent", "opposite"):
        s["target"] = DEFAULT_TARGET
    return s


def _clean_rules(rules) -> list[dict]:
    out = []
    for rule in rules:
        if isinstance(rule, str):
            rule = {"id": rule}
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id", "")).strip().lower()
        if rid not in KNOWN_RULE_IDS:
            continue
        params = rule.get("params") if isinstance(rule.get("params"), dict) else {}
        out.append({"id": rid, "params": params})
    return out


_ENTRY_IDS = {
    "ema_cross_up", "sma_cross_up", "macd_cross_up", "rsi_cross_up",
    "bb_lower_touch", "support_bounce", "breakout_high", "supertrend_up",
    "ichimoku_cloud_up", "breakdown_low",
}
_EXIT_IDS = {
    "ema_cross_down", "sma_cross_down", "macd_cross_down", "rsi_cross_down",
    "bb_upper_touch", "supertrend_down", "resistance_reject",
    "ichimoku_cloud_down",
}
_FILTER_IDS = {
    "rsi_below", "rsi_above", "price_above_ema", "price_below_ema",
    "vwap_above", "vwap_below", "adx_above", "volume_spike", "session",
    "ema_stack_bull", "ema_stack_bear",
    "ichimoku_above_cloud", "ichimoku_below_cloud",
}

ALL_BUCKETS = ("entry", "exit", "short_entry", "short_exit", "filters", "short_filters")


def apply_patch(strategy: dict, patch: dict) -> dict:
    """Merge a patch (e.g. from the learning loop) into the strategy."""
    s = json.loads(json.dumps(strategy))
    for bucket in ALL_BUCKETS:
        add = (patch.get("rules") or {}).get(bucket)
        if isinstance(add, list):
            s["rules"][bucket] = _clean_rules(s["rules"].get(bucket, []) + add)
    if isinstance(patch.get("rules"), dict) and isinstance(patch["rules"].get("remove"), list):
        for rid in patch["rules"]["remove"]:
            for bucket in ALL_BUCKETS:
                s["rules"][bucket] = [r for r in s["rules"][bucket] if r["id"] != rid]
    if isinstance(patch.get("stop"), dict):
        cur = s["stop"] or dict(DEFAULT_STOP)
        cur.setdefault("params", {}).update(patch["stop"].get("params", {}))
        if patch["stop"].get("type"):
            cur["type"] = patch["stop"]["type"]
        s["stop"] = cur
    if isinstance(patch.get("target"), dict):
        cur = s["target"] or dict(DEFAULT_TARGET)
        cur.setdefault("params", {}).update(patch["target"].get("params", {}))
        if patch["target"].get("type"):
            cur["type"] = patch["target"]["type"]
        s["target"] = cur
    if patch.get("timeframe"):
        s["timeframe"] = str(patch["timeframe"])
    if patch.get("side") in ("long", "short", "both"):
        s["side"] = patch["side"]
    s["revision"] = int(s.get("revision", 1)) + 1
    s["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return normalize_strategy(s, revision=s["revision"])


# ---------------------------------------------------------------------------
# Pine code builders
# ---------------------------------------------------------------------------

class _Ctx:
    """Collects declarations / plots while rule builders emit expressions.

    Dedupes identical series (e.g. the same EMA 9 used by both an entry and
    an exit rule) and identical plot titles — both would be Pine compile
    errors / warnings otherwise.
    """

    def __init__(self):
        self.decls: list[str] = []
        self.plots: list[str] = []
        self._series: dict[str, str] = {}   # expression → var name (or tuple of names)
        self._plot_keys: set[str] = set()
        self._n = 0

    def series(self, expression: str, base: str) -> str:
        if expression in self._series:
            return self._series[expression]
        self._n += 1
        name = f"{base}{self._n}"
        self._series[expression] = name
        self.decls.append(f"{name} = {expression}")
        return name

    def tuple_series(self, expression: str, base: str, arity: int) -> tuple:
        if expression in self._series:
            return self._series[expression]
        self._n += 1
        names = tuple(f"{base}{self._n}{chr(97 + i)}" for i in range(arity))
        self._series[expression] = names
        self.decls.append(f"[{', '.join(names)}] = {expression}")
        return names

    def plot(self, expr, title, color="#888888", style=None, linewidth=None):
        if title in self._plot_keys:
            return
        self._plot_keys.add(title)
        extras = ""
        if style:
            extras += f", style={style}"
        if linewidth:
            extras += f", linewidth={linewidth}"
        self.plots.append(f"plot({expr}, '{title}', color={color}{extras})")


def _int(p, key, default):
    try:
        return int(float(p.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _float(p, key, default):
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _rule_ema(ctx, p, up: bool) -> str:
    fast, slow = _int(p, "fast", 9), _int(p, "slow", 21)
    f = ctx.series(f"ta.ema(close, {fast})", "emaFast")
    s = ctx.series(f"ta.ema(close, {slow})", "emaSlow")
    ctx.plot(f, f"EMA {fast}", color="color.new(#26a69a, 0)", linewidth=1)
    ctx.plot(s, f"EMA {slow}", color="color.new(#ff9800, 0)", linewidth=1)
    return f"ta.crossover({f}, {s})" if up else f"ta.crossunder({f}, {s})"


def _rule_sma(ctx, p, up: bool) -> str:
    fast, slow = _int(p, "fast", 10), _int(p, "slow", 30)
    f = ctx.series(f"ta.sma(close, {fast})", "smaFast")
    s = ctx.series(f"ta.sma(close, {slow})", "smaSlow")
    ctx.plot(f, f"SMA {fast}", color="color.new(#26a69a, 0)", linewidth=1)
    ctx.plot(s, f"SMA {slow}", color="color.new(#ff9800, 0)", linewidth=1)
    return f"ta.crossover({f}, {s})" if up else f"ta.crossunder({f}, {s})"


def _rule_macd(ctx, p, up: bool) -> str:
    fast, slow, smooth = _int(p, "fast", 12), _int(p, "slow", 26), _int(p, "smooth", 9)
    m, sig, hist = ctx.tuple_series(f"ta.macd(close, {fast}, {slow}, {smooth})", "macd", 3)
    ctx.plot(hist, "MACD hist", color="color.new(#888888, 0)", style="plot.style_columns")
    return f"ta.crossover({m}, {sig})" if up else f"ta.crossunder({m}, {sig})"


def _rule_rsi_cross(ctx, p, up: bool) -> str:
    length, level = _int(p, "length", 14), _float(p, "level", 50)
    r = ctx.series(f"ta.rsi(close, {length})", "rsi")
    return f"ta.crossover({r}, {level})" if up else f"ta.crossunder({r}, {level})"


def _rule_bb_touch(ctx, p, lower: bool) -> str:
    length, mult = _int(p, "length", 20), _float(p, "mult", 2.0)
    b, bUp, bLow = ctx.tuple_series(f"ta.bb(close, {length}, {mult})", "bb", 3)
    ctx.plot(bUp, "BB upper", color="color.new(#aaaaaa, 60)")
    ctx.plot(bLow, "BB lower", color="color.new(#aaaaaa, 60)")
    return f"low <= {bLow}" if lower else f"high >= {bUp}"


def _rule_supertrend(ctx, p, up: bool) -> str:
    factor, atr_period = _float(p, "factor", 3.0), _int(p, "atr", 10)
    st, dr = ctx.tuple_series(f"ta.supertrend({factor}, {atr_period})", "st", 2)
    ctx.plot(
        st, "Supertrend",
        color=f"{dr} < 0 ? color.new(#26a69a, 0) : color.new(#ef5350, 0)",
    )
    return f"{dr} < 0" if up else f"{dr} > 0"


def _rule_breakout(ctx, p) -> str:
    lookback = _int(p, "lookback", 20)
    return f"high > ta.highest(high, {lookback})[1]"


def _rule_breakdown(ctx, p) -> str:
    lookback = _int(p, "lookback", 20)
    return f"low < ta.lowest(low, {lookback})[1]"


def _rule_support_bounce(ctx, p) -> str:
    left, right = _int(p, "left", 10), _int(p, "right", 10)
    pl = ctx.series(f"ta.pivotlow(low, {left}, {right})", "pivotLow")
    ctx.plot(pl, "Pivot low", color="color.new(#ef5350, 0)", style="plot.style_circles")
    return f"not na({pl}) and low <= {pl} and close > open"


def _rule_resistance_reject(ctx, p) -> str:
    left, right = _int(p, "left", 10), _int(p, "right", 10)
    ph = ctx.series(f"ta.pivothigh(high, {left}, {right})", "pivotHigh")
    ctx.plot(ph, "Pivot high", color="color.new(#26a69a, 0)", style="plot.style_circles")
    return f"not na({ph}) and high >= {ph} and close < open"


def _ichimoku(ctx, p) -> tuple[str, str]:
    conv = _int(p, "conversion", 9)
    base = _int(p, "base", 26)
    lag = _int(p, "lagging", 52)
    disp = _int(p, "disp", 26)
    convL, baseL, lead1, lead2, lagL = ctx.tuple_series(
        f"ta.ichimoku({conv}, {base}, {lag}, {disp})", "ichi", 5
    )
    ctx.plot(f"math.max({lead1}[{disp}], {lead2}[{disp}])", "Ichimoku cloud top",
             color="color.new(#26a69a, 70)")
    ctx.plot(f"math.min({lead1}[{disp}], {lead2}[{disp}])", "Ichimoku cloud bottom",
             color="color.new(#ef5350, 70)")
    ctx.plot(convL, "Ichimoku conversion", color="color.new(#42a5f5, 0)", linewidth=1)
    ctx.plot(baseL, "Ichimoku base", color="color.new(#ab47bc, 0)", linewidth=1)
    return (f"math.max({lead1}[{disp}], {lead2}[{disp}])",
            f"math.min({lead1}[{disp}], {lead2}[{disp}])")


def _rule_ichimoku_cloud(ctx, p, up: bool) -> str:
    top, bot = _ichimoku(ctx, p)
    return f"ta.crossover(close, {top})" if up else f"ta.crossunder(close, {bot})"


def _rule_ichimoku_side(ctx, p, above: bool) -> str:
    top, bot = _ichimoku(ctx, p)
    return f"close > {top}" if above else f"close < {bot}"


def _rule_ema_stack(ctx, p, bull: bool) -> str:
    fast, mid, slow = _int(p, "fast", 10), _int(p, "mid", 20), _int(p, "slow", 50)
    f = ctx.series(f"ta.ema(close, {fast})", "stackFast")
    m = ctx.series(f"ta.ema(close, {mid})", "stackMid")
    s = ctx.series(f"ta.ema(close, {slow})", "stackSlow")
    ctx.plot(f, f"Stack EMA {fast}", color="color.new(#26a69a, 0)", linewidth=1)
    ctx.plot(m, f"Stack EMA {mid}", color="color.new(#ff9800, 0)", linewidth=1)
    ctx.plot(s, f"Stack EMA {slow}", color="color.new(#ef5350, 0)", linewidth=1)
    return f"{f} > {m} and {m} > {s}" if bull else f"{f} < {m} and {m} < {s}"


def _rule_rsi_filter(ctx, p, below: bool) -> str:
    length, level = _int(p, "length", 14), _float(p, "level", 50)
    r = ctx.series(f"ta.rsi(close, {length})", "rsi")
    return f"{r} < {level}" if below else f"{r} > {level}"


def _rule_price_ema(ctx, p, above: bool) -> str:
    length = _int(p, "length", 200)
    e = ctx.series(f"ta.ema(close, {length})", "emaTrend")
    ctx.plot(e, f"EMA {length} filter", color="color.new(#42a5f5, 0)", linewidth=1)
    return f"close > {e}" if above else f"close < {e}"


def _rule_vwap(ctx, p, above: bool) -> str:
    v = ctx.series("ta.vwap(close)", "vwap")
    ctx.plot(v, "VWAP", color="color.new(#ab47bc, 0)", linewidth=1)
    return f"close > {v}" if above else f"close < {v}"


def _rule_adx(ctx, p) -> str:
    length, level = _int(p, "length", 14), _float(p, "level", 25)
    d, m, a = ctx.tuple_series(f"ta.dmi({length}, {length})", "dmi", 3)
    return f"{a} > {level}"


def _rule_volume(ctx, p) -> str:
    length, mult = _int(p, "length", 20), _float(p, "mult", 2.0)
    return f"volume > ta.sma(volume, {length}) * {mult}"


def _rule_session(ctx, p) -> str:
    tz = str(p.get("tz", "America/New_York"))
    start, end = _int(p, "start", 9 * 60), _int(p, "end", 17 * 60)
    t = ctx.series(f'hour(time, "{tz}") * 60 + minute(time, "{tz}")', "mins")
    return f"{t} >= {start} and {t} < {end}"


_RULES = {
    "ema_cross_up": lambda c, p: _rule_ema(c, p, True),
    "ema_cross_down": lambda c, p: _rule_ema(c, p, False),
    "sma_cross_up": lambda c, p: _rule_sma(c, p, True),
    "sma_cross_down": lambda c, p: _rule_sma(c, p, False),
    "macd_cross_up": lambda c, p: _rule_macd(c, p, True),
    "macd_cross_down": lambda c, p: _rule_macd(c, p, False),
    "rsi_cross_up": lambda c, p: _rule_rsi_cross(c, p, True),
    "rsi_cross_down": lambda c, p: _rule_rsi_cross(c, p, False),
    "bb_lower_touch": lambda c, p: _rule_bb_touch(c, p, True),
    "bb_upper_touch": lambda c, p: _rule_bb_touch(c, p, False),
    "supertrend_up": lambda c, p: _rule_supertrend(c, p, True),
    "supertrend_down": lambda c, p: _rule_supertrend(c, p, False),
    "breakout_high": _rule_breakout,
    "breakdown_low": _rule_breakdown,
    "support_bounce": _rule_support_bounce,
    "resistance_reject": _rule_resistance_reject,
    "ichimoku_cloud_up": lambda c, p: _rule_ichimoku_cloud(c, p, True),
    "ichimoku_cloud_down": lambda c, p: _rule_ichimoku_cloud(c, p, False),
    "ichimoku_above_cloud": lambda c, p: _rule_ichimoku_side(c, p, True),
    "ichimoku_below_cloud": lambda c, p: _rule_ichimoku_side(c, p, False),
    "rsi_below": lambda c, p: _rule_rsi_filter(c, p, True),
    "rsi_above": lambda c, p: _rule_rsi_filter(c, p, False),
    "price_above_ema": lambda c, p: _rule_price_ema(c, p, True),
    "price_below_ema": lambda c, p: _rule_price_ema(c, p, False),
    "vwap_above": lambda c, p: _rule_vwap(c, p, True),
    "vwap_below": lambda c, p: _rule_vwap(c, p, False),
    "adx_above": _rule_adx,
    "volume_spike": _rule_volume,
    "session": _rule_session,
    "ema_stack_bull": lambda c, p: _rule_ema_stack(c, p, True),
    "ema_stack_bear": lambda c, p: _rule_ema_stack(c, p, False),
}


def _rule_label(rule: dict) -> str:
    p = rule.get("params") or {}
    return f"{rule['id']} {json.dumps(p, sort_keys=True)}"


def summarize_rules(strategy: dict) -> str:
    """Human-readable strategy summary for chat replies and code headers."""
    lines = []
    side = strategy.get("side", "long")
    rules = strategy.get("rules", {})
    if side in ("long", "both"):
        lines.append("Long entry: " + (", ".join(_rule_label(r) for r in rules.get("entry", [])) or "—"))
        lines.append("Long exit: " + (", ".join(_rule_label(r) for r in rules.get("exit", [])) or "—"))
    if side in ("short", "both"):
        lines.append("Short entry: " + (", ".join(_rule_label(r) for r in rules.get("short_entry", [])) or "—"))
        lines.append("Short exit: " + (", ".join(_rule_label(r) for r in rules.get("short_exit", [])) or "—"))
    if side in ("long", "both"):
        lines.append("Filters: " + (", ".join(_rule_label(r) for r in rules.get("filters", [])) or "—"))
    if side in ("short", "both"):
        lines.append("Short filters: " + (", ".join(_rule_label(r) for r in rules.get("short_filters", [])) or "—"))
    stop = strategy.get("stop")
    target = strategy.get("target")
    if stop:
        lines.append(f"Stop: {stop['type']} {json.dumps(stop.get('params') or {}, sort_keys=True)}")
    if target:
        lines.append(f"Target: {target['type']} {json.dumps(target.get('params') or {}, sort_keys=True)}")
    return "\n".join(lines)


def _build_side(ctx, rules: dict, bucket_entry: str, bucket_exit: str, bucket_filters: str):
    entry = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in rules[bucket_entry]])
    exit_ = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in rules[bucket_exit]])
    filters = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in rules[bucket_filters]])
    return entry, exit_, filters


def _join(exprs) -> str:
    exprs = [e for e in exprs if e]
    if not exprs:
        return "false"
    if len(exprs) == 1:
        return exprs[0]
    return "(" + " and ".join(exprs) + ")"


def _stop_plan(ctx, strategy: dict, var: str, direction: str) -> list[str]:
    """Return assignment lines for the stop of one side (series helpers are
    hoisted into ctx automatically, since Pine forbids local declarations)."""
    stop = strategy.get("stop")
    if not stop:
        return [f"{var} := na"]
    p = stop.get("params") or {}
    if stop["type"] == "atr":
        op = "+" if direction == "short" else "-"
        return [f"{var} := close {op} ta.atr({_int(p, 'length', 14)}) * {_float(p, 'mult', 2.0)}"]
    if stop["type"] == "percent":
        sign = "+" if direction == "short" else "-"
        return [f"{var} := close * (1 {sign} {_float(p, 'pct', 1.0)} / 100.0)"]
    if stop["type"] == "swing":
        left, right = _int(p, "left", 10), _int(p, "right", 10)
        if direction == "short":
            ph = ctx.series(f"ta.pivothigh(high, {left}, {right})", "swingStopHigh")
            return [f"{var} := na({ph}) ? close * 1.01 : {ph}"]
        pl = ctx.series(f"ta.pivotlow(low, {left}, {right})", "swingStopLow")
        return [f"{var} := na({pl}) ? close * 0.99 : {pl}"]
    return [f"{var} := na"]


def _target_assign(strategy: dict, var: str, entry: str, stop: str, direction: str) -> str:
    target = strategy.get("target")
    if not target:
        return f"{var} := na"
    p = target.get("params") or {}
    if target["type"] == "rr":
        rr = _float(p, "rr", 2.0)
        if direction == "long":
            return f"{var} := {entry} + ({entry} - {stop}) * {rr}"
        return f"{var} := {entry} - ({stop} - {entry}) * {rr}"
    if target["type"] == "percent":
        sign = "-" if direction == "short" else "+"
        return f"{var} := {entry} * (1 {sign} {_float(p, 'pct', 2.0)} / 100.0)"
    if target["type"] == "opposite":
        # distance mirrored on the other side of entry
        if direction == "long":
            return f"{var} := {entry} + ({entry} - {stop})"
        return f"{var} := {entry} - ({stop} - {entry})"
    return f"{var} := na"


def _safe_name(name: str) -> str:
    """Strip characters that would break the generated Pine string literals."""
    return re.sub(r"['\"\\\n\r\t]", "", str(name))[:40] or "Strategy"


def _header(strategy: dict) -> str:
    src = strategy.get("source_video") or "hand-authored"
    rules = summarize_rules(strategy).replace("\n", "\n//  ")
    return (
        f"// ════════════════════════════════════════════════════════════════\n"
        f"//  {_safe_name(strategy['name'])}  (revision {strategy.get('revision', 1)})\n"
        f"//  Generated by the Trading Brain — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"//  Source: {src}\n"
        f"//  Market: {strategy.get('market', 'any')} · timeframe {strategy.get('timeframe', 'any')} · side {strategy.get('side', 'long')}\n"
        f"//  {rules}\n"
        f"//  NOTE: educational tool — not financial advice. Test before live use.\n"
        f"// ════════════════════════════════════════════════════════════════"
    )


def _alert_messages(strategy: dict) -> tuple[str, str, str, str]:
    """Return (buy, sell, short, cover) alert message strings."""
    name = _safe_name(strategy["name"])
    buy = (
        f'{{"strategy":"{name}","revision":{strategy.get("revision", 1)},'
        f'"action":"BUY","side":"long","symbol":"{{{{ticker}}}}","interval":"{{{{interval}}}}",'
        f'"price":{{{{close}}}},"time":"{{{{time}}}}"}}'
    )
    sell = buy.replace('"action":"BUY"', '"action":"SELL"')
    short = buy.replace('"action":"BUY"', '"action":"SELLSHORT"').replace('"side":"long"', '"side":"short"')
    cover = sell.replace('"action":"SELL"', '"action":"BUYTOCOVER"').replace('"side":"long"', '"side":"short"')
    return buy, sell, short, cover


def generate_indicator(strategy: dict) -> str:
    """Compile the strategy into a Pine v6 ``indicator()`` script with alerts."""
    strategy = normalize_strategy(strategy)
    side = strategy.get("side", "long")
    do_long = side in ("long", "both")
    do_short = side in ("short", "both")
    name = _safe_name(strategy["name"])

    ctx = _Ctx()
    buy_msg, sell_msg, short_msg, cover_msg = _alert_messages(strategy)

    long_cond = long_exit = "false"
    short_cond = short_exit = "false"
    if do_long:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "entry", "exit", "filters")
        long_cond = _join([e for e in (entry, filters) if e != "false"])
        long_exit = exit_
    if do_short:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "short_entry", "short_exit", "short_filters")
        short_cond = _join([e for e in (entry, filters) if e != "false"])
        short_exit = exit_

    stop = strategy.get("stop")
    target = strategy.get("target")
    has_stop = stop is not None
    has_target = target is not None and target.get("type") != "opposite"

    long_stop_assigns = _stop_plan(ctx, strategy, "longStop", "long")
    short_stop_assigns = _stop_plan(ctx, strategy, "shortStop", "short")
    long_target_assign = _target_assign(strategy, "longTarget", "longEntry", "longStop", "long")
    short_target_assign = _target_assign(strategy, "shortTarget", "shortEntry", "shortStop", "short")

    long_guard = ""
    if has_stop:
        long_guard += " or (inLong and not na(longStop) and low <= longStop)"
    if has_target:
        long_guard += " or (inLong and not na(longTarget) and high >= longTarget)"
    short_guard = ""
    if has_stop:
        short_guard += " or (inShort and not na(shortStop) and high >= shortStop)"
    if has_target:
        short_guard += " or (inShort and not na(shortTarget) and low <= shortTarget)"
    if long_exit == "false" and not has_stop and not has_target:
        long_guard += " or bar_index - longBar >= maxBarsInTrade"
    if short_exit == "false" and not has_stop and not has_target:
        short_guard += " or bar_index - shortBar >= maxBarsInTrade"

    decls = "\n".join(ctx.decls) or "// (no indicator declarations)"
    plots = "\n".join(ctx.plots)

    L = []
    L.append("//@version=6")
    L.append(f'indicator("{name} [Brain]", shorttitle="Brain {_slug(name)}", overlay=true, max_labels_count=500)')
    L.append("")
    L.append(_header(strategy))
    L.append("")
    L.append("// ── Inputs ─────────────────────────────────────────────────────────")
    L.append("maxBarsInTrade = input.int(100, 'Max bars in trade', minval=5)")
    L.append("")
    L.append("// ── Indicator computations ────────────────────────────────────────")
    L.append(decls)
    L.append("")
    L.append("// ── Trade state ───────────────────────────────────────────────────")
    if do_long:
        L += [
            "var bool  inLong     = false",
            "var float longEntry  = na",
            "var float longStop   = na",
            "var float longTarget = na",
            "var int   longBar    = na",
        ]
    if do_short:
        L += [
            "var bool  inShort     = false",
            "var float shortEntry  = na",
            "var float shortStop   = na",
            "var float shortTarget = na",
            "var int   shortBar    = na",
        ]
    L.append("")
    L.append("// ── Conditions ────────────────────────────────────────────────────")
    if do_long:
        L += [
            f"longCond  = {long_cond}",
            f"longExitCond = {long_exit}",
            "longTrigger = longCond and not inLong",
            f"longExitTrigger = inLong and (longExitCond{long_guard})",
        ]
    if do_short:
        L += [
            f"shortCond  = {short_cond}",
            f"shortExitCond = {short_exit}",
            "shortTrigger = shortCond and not inShort",
            f"shortExitTrigger = inShort and (shortExitCond{short_guard})",
        ]
    L.append("")
    L.append("// ── Position engine ───────────────────────────────────────────────")
    if do_long and do_short:
        L += [
            "// flip: long signal while short → cover first, then enter long",
            "if longTrigger and inShort",
            "    inShort := false",
            "    shortEntry := na",
            "    shortBar := na",
            "    shortStop := na",
            "    shortTarget := na",
            f"    alert({cover_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "// flip: short signal while long → sell first, then enter short",
            "if shortTrigger and inLong",
            "    inLong := false",
            "    longEntry := na",
            "    longBar := na",
            "    longStop := na",
            "    longTarget := na",
            f"    alert({sell_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    if do_long:
        L += [
            "if longTrigger",
            "    inLong := true",
            "    longEntry := close",
            "    longBar := bar_index",
            *[f"    {a}" for a in long_stop_assigns],
            f"    {long_target_assign}",
            f"    alert({buy_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "if longExitTrigger",
            "    inLong := false",
            "    longEntry := na",
            "    longBar := na",
            "    longStop := na",
            "    longTarget := na",
            f"    alert({sell_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    if do_short:
        L += [
            "if shortTrigger",
            "    inShort := true",
            "    shortEntry := close",
            "    shortBar := bar_index",
            *[f"    {a}" for a in short_stop_assigns],
            f"    {short_target_assign}",
            f"    alert({short_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "if shortExitTrigger",
            "    inShort := false",
            "    shortEntry := na",
            "    shortBar := na",
            "    shortStop := na",
            "    shortTarget := na",
            f"    alert({cover_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    L.append("// ── Visuals ───────────────────────────────────────────────────────")
    if do_long:
        L.append("plotshape(longTrigger, 'BUY', style=shape.triangleup, location=location.belowbar, color=color.new(#26a69a, 0), size=size.small)")
        L.append("plotshape(longExitTrigger, 'SELL', style=shape.triangledown, location=location.abovebar, color=color.new(#ef5350, 0), size=size.small)")
    if do_short:
        L.append("plotshape(shortTrigger, 'SHORT', style=shape.triangledown, location=location.abovebar, color=color.new(#ff7043, 0), size=size.small)")
        L.append("plotshape(shortExitTrigger, 'COVER', style=shape.triangleup, location=location.belowbar, color=color.new(#42a5f5, 0), size=size.small)")
    if do_long and do_short:
        L.append("bgcolor(inLong ? color.new(#26a69a, 92) : inShort ? color.new(#ef5350, 92) : na)")
    elif do_long:
        L.append("bgcolor(inLong ? color.new(#26a69a, 92) : na)")
    else:
        L.append("bgcolor(inShort ? color.new(#ef5350, 92) : na)")
    if do_long and has_stop:
        L.append("plot(inLong ? longStop : na, 'Long stop', style=plot.style_linebr, color=color.new(#ef5350, 20), linewidth=2)")
    if do_long and has_target:
        L.append("plot(inLong ? longTarget : na, 'Long target', style=plot.style_linebr, color=color.new(#26a69a, 20), linewidth=2)")
    if do_short and has_stop:
        L.append("plot(inShort ? shortStop : na, 'Short stop', style=plot.style_linebr, color=color.new(#ff7043, 20), linewidth=2)")
    if do_short and has_target:
        L.append("plot(inShort ? shortTarget : na, 'Short target', style=plot.style_linebr, color=color.new(#42a5f5, 20), linewidth=2)")
    L.append("")
    if plots:
        L.append("// ── Indicator lines ───────────────────────────────────────────────")
        L.append(plots)
        L.append("")
    L.append("// ── Alerts → brain webhook ────────────────────────────────────────")
    if do_long:
        L.append(f'alertcondition(longCond, "Brain BUY — {name}", message={buy_msg!r})')
        L.append(f'alertcondition(longExitCond, "Brain SELL — {name}", message={sell_msg!r})')
    if do_short:
        L.append(f'alertcondition(shortCond, "Brain SHORT — {name}", message={short_msg!r})')
        L.append(f'alertcondition(shortExitCond, "Brain COVER — {name}", message={cover_msg!r})')
    return "\n".join(L) + "\n"


def generate_strategy(strategy: dict) -> str:
    """Compile the strategy into a Pine v6 ``strategy()`` backtest script."""
    strategy = normalize_strategy(strategy)
    side = strategy.get("side", "long")
    do_long = side in ("long", "both")
    do_short = side in ("short", "both")
    name = _safe_name(strategy["name"])

    ctx = _Ctx()

    long_cond = long_exit = "false"
    short_cond = short_exit = "false"
    if do_long:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "entry", "exit", "filters")
        long_cond = _join([e for e in (entry, filters) if e != "false"])
        long_exit = exit_
    if do_short:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "short_entry", "short_exit", "short_filters")
        short_cond = _join([e for e in (entry, filters) if e != "false"])
        short_exit = exit_

    stop = strategy.get("stop")
    target = strategy.get("target")
    has_stop = stop is not None
    has_target = target is not None and target.get("type") != "opposite"

    long_stop_assigns = _stop_plan(ctx, strategy, "longStop", "long")
    short_stop_assigns = _stop_plan(ctx, strategy, "shortStop", "short")
    long_target_assign = _target_assign(strategy, "longTarget", "longEntry", "longStop", "long")
    short_target_assign = _target_assign(strategy, "shortTarget", "shortEntry", "shortStop", "short")

    L = []
    L.append("//@version=6")
    L.append(
        f'strategy("{name} [Brain BT]", shorttitle="Brain {_slug(name)} BT", '
        f'overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, '
        f'default_qty_value=100, commission_type=strategy.commission.percent, '
        f'commission_value=0.1, pyramiding=0)'
    )
    L.append("")
    L.append(_header(strategy))
    L.append("")
    L.append("// ── Indicator computations ────────────────────────────────────────")
    L.extend(ctx.decls)
    L.append("")
    L.append("// ── Position state (stop / target captured at entry) ─────────────")
    if do_long:
        L.append("var float longEntry = na")
        L.append("var float longStop = na")
        L.append("var float longTarget = na")
    if do_short:
        L.append("var float shortEntry = na")
        L.append("var float shortStop = na")
        L.append("var float shortTarget = na")
    L.append("")
    L.append("// ── Conditions ────────────────────────────────────────────────────")
    if do_long:
        L.append(f"longCond  = {long_cond}")
        L.append(f"longExitCond = {long_exit}")
    if do_short:
        L.append(f"shortCond  = {short_cond}")
        L.append(f"shortExitCond = {short_exit}")
    L.append("")
    L.append("// ── Entries ───────────────────────────────────────────────────────")
    if do_long:
        L += [
            "if longCond and strategy.position_size <= 0",
            "    strategy.entry('Long', strategy.long)",
            "    longEntry := close",
            *[f"    {a}" for a in long_stop_assigns],
            f"    {long_target_assign}",
            "",
        ]
    if do_short:
        L += [
            "if shortCond and strategy.position_size >= 0",
            "    strategy.entry('Short', strategy.short)",
            "    shortEntry := close",
            *[f"    {a}" for a in short_stop_assigns],
            f"    {short_target_assign}",
            "",
        ]
    L.append("// ── Signal exits ───────────────────────────────────────────────────")
    if do_long:
        L += [
            "if longExitCond and strategy.position_size > 0",
            "    strategy.close('Long', comment='Signal exit')",
            "",
        ]
    if do_short:
        L += [
            "if shortExitCond and strategy.position_size < 0",
            "    strategy.close('Short', comment='Signal exit')",
            "",
        ]
    L.append("// ── Risk management exits ─────────────────────────────────────────")
    if do_long and (has_stop or has_target):
        parts, guards = [], ["not na(longStop)"]
        if has_stop:
            parts.append("stop=longStop")
        if has_target:
            parts.append("limit=longTarget")
            guards.append("not na(longTarget)")
        L.append(f"if strategy.position_size > 0 and {' and '.join(guards)}")
        L.append(f"    strategy.exit('LongRisk', from_entry='Long', {', '.join(parts)})")
    elif do_long:
        L.append("// (no long stop/target configured)")
    if do_short and (has_stop or has_target):
        parts, guards = [], ["not na(shortStop)"]
        if has_stop:
            parts.append("stop=shortStop")
        if has_target:
            parts.append("limit=shortTarget")
            guards.append("not na(shortTarget)")
        L.append(f"if strategy.position_size < 0 and {' and '.join(guards)}")
        L.append(f"    strategy.exit('ShortRisk', from_entry='Short', {', '.join(parts)})")
    elif do_short:
        L.append("// (no short stop/target configured)")
    L.append("")
    if ctx.plots:
        L.append("// ── Indicator lines ───────────────────────────────────────────────")
        L.extend(ctx.plots)
    return "\n".join(L) + "\n"


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", name)[:20]
    return slug or "Strategy"
