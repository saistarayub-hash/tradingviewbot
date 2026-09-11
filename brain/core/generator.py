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
    "halftrend_up", "halftrend_down", "halftrend_bull", "halftrend_bear",
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
    "halftrend_up": "halftrend_down", "halftrend_down": "halftrend_up",
    "halftrend_bull": "halftrend_bear", "halftrend_bear": "halftrend_bull",
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
    raw, name: str | None = None, source_video: str | None = None, revision: int | None = None
) -> dict:
    """Coerce a raw extraction (LLM JSON or heuristics dict) into canonical form."""
    if revision is None and isinstance(raw, dict):
        try:
            revision = int(raw.get("revision", 1))
        except (TypeError, ValueError):
            revision = 1
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
    seen: set[str] = set()
    for rule in rules:
        if isinstance(rule, str):
            rule = {"id": rule}
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id", "")).strip().lower()
        if rid not in KNOWN_RULE_IDS or rid in seen:
            continue
        seen.add(rid)
        params = rule.get("params") if isinstance(rule.get("params"), dict) else {}
        out.append({"id": rid, "params": params})
    return out


_ENTRY_IDS = {
    "ema_cross_up", "sma_cross_up", "macd_cross_up", "rsi_cross_up",
    "bb_lower_touch", "support_bounce", "breakout_high", "supertrend_up",
    "ichimoku_cloud_up", "breakdown_low", "halftrend_up",
}
_EXIT_IDS = {
    "ema_cross_down", "sma_cross_down", "macd_cross_down", "rsi_cross_down",
    "bb_upper_touch", "supertrend_down", "resistance_reject",
    "ichimoku_cloud_down", "halftrend_down",
}
_FILTER_IDS = {
    "rsi_below", "rsi_above", "price_above_ema", "price_below_ema",
    "vwap_above", "vwap_below", "adx_above", "volume_spike", "session",
    "ema_stack_bull", "ema_stack_bear",
    "ichimoku_above_cloud", "ichimoku_below_cloud",
    "halftrend_bull", "halftrend_bear",
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


def _half_trend_engine(ctx, p) -> tuple[str, str, str, str]:
    """Append the HalfTrend engine (trend matrix + baseline + bands).

    Credited to everget; adapted from BigBeluga's HalfTrend Signal Engine
    (CC BY-NC-SA 4.0). Returns (trend, line, bandHigh, bandLow) series names.
    """
    amplitude = _int(p, "amplitude", 20)
    deviation = _float(p, "channel_deviation", 2.0)
    key = f"halftrend:{amplitude}:{deviation}"
    if key in ctx._series:
        return ctx._series[key]
    ctx._n += 1
    i = ctx._n
    trend, next_t = f"htTrend{i}", f"htNext{i}"
    max_low, min_high = f"htMaxLow{i}", f"htMinHigh{i}"
    up, down = f"htUp{i}", f"htDown{i}"
    atr2, dev = f"htAtr{i}", f"htDev{i}"
    hi_px, lo_px = f"htHi{i}", f"htLo{i}"
    hi_ma, lo_ma = f"htHiMa{i}", f"htLoMa{i}"
    line, atr_hi, atr_lo = f"htLine{i}", f"htBandHi{i}", f"htBandLo{i}"
    ctx.decls.extend([
        f"var int {trend} = 0",
        f"var int {next_t} = 0",
        f"var float {max_low} = low",
        f"var float {min_high} = high",
        f"var float {up} = 0.0",
        f"var float {down} = 0.0",
        "",
        f"// — HalfTrend engine (amplitude {amplitude}, channel deviation {deviation})",
        f"{atr2} = ta.atr(100) / 2",
        f"{dev} = {deviation} * {atr2}",
        f"{hi_px} = high[math.abs(ta.highestbars(high, {amplitude}))]",
        f"{lo_px} = low[math.abs(ta.lowestbars(low, {amplitude}))]",
        f"{hi_ma} = ta.sma(high, {amplitude})",
        f"{lo_ma} = ta.sma(low, {amplitude})",
        f"{atr_hi} = 0.0",
        f"{atr_lo} = 0.0",
        f"if {next_t} == 1",
        f"    {max_low} := math.max({lo_px}, {max_low})",
        f"    if {hi_ma} < {max_low} and close < nz(low[1], low)",
        f"        {trend} := 1",
        f"        {next_t} := 0",
        f"        {min_high} := {hi_px}",
        f"else",
        f"    {min_high} := math.min({hi_px}, {min_high})",
        f"    if {lo_ma} > {min_high} and close > nz(high[1], high)",
        f"        {trend} := 0",
        f"        {next_t} := 1",
        f"        {max_low} := {lo_px}",
        f"if {trend} == 0",
        f"    if not na({trend}[1]) and {trend}[1] != 0",
        f"        {up} := na({down}[1]) ? {down} : {down}[1]",
        f"    else",
        f"        {up} := na({up}[1]) ? {max_low} : math.max({max_low}, {up}[1])",
        f"    {atr_hi} := {up} + {dev}",
        f"    {atr_lo} := {up} - {dev}",
        f"else",
        f"    if not na({trend}[1]) and {trend}[1] != 1",
        f"        {down} := na({up}[1]) ? {up} : {up}[1]",
        f"    else",
        f"        {down} := na({down}[1]) ? {min_high} : math.min({min_high}, {down}[1])",
        f"    {atr_hi} := {down} + {dev}",
        f"    {atr_lo} := {down} - {dev}",
        f"{line} = {trend} == 0 ? {up} : {down}",
    ])
    names = (trend, line, atr_hi, atr_lo)
    ctx._series[key] = names
    ctx.plot(line, "HalfTrend",
             color=f"{trend} == 0 ? color.new(#26a69a, 0) : color.new(#ef5350, 0)",
             linewidth=3)
    ctx.plot(atr_hi, "HalfTrend band high", color="color.new(#ef5350, 78)")
    ctx.plot(atr_lo, "HalfTrend band low", color="color.new(#26a69a, 78)")
    return names


def _rule_halftrend(ctx, p, up: bool) -> str:
    trend, _line, _hi, _lo = _half_trend_engine(ctx, p)
    return f"{trend} == 0 and {trend}[1] == 1" if up else f"{trend} == 1 and {trend}[1] == 0"


def _rule_halftrend_filter(ctx, p, bull: bool) -> str:
    trend, _line, _hi, _lo = _half_trend_engine(ctx, p)
    return f"{trend} == 0" if bull else f"{trend} == 1"


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
    "halftrend_up": lambda c, p: _rule_halftrend(c, p, True),
    "halftrend_down": lambda c, p: _rule_halftrend(c, p, False),
    "halftrend_bull": lambda c, p: _rule_halftrend_filter(c, p, True),
    "halftrend_bear": lambda c, p: _rule_halftrend_filter(c, p, False),
}


def _rule_label(rule: dict) -> str:
    p = rule.get("params") or {}
    return f"{rule['id']} {json.dumps(p, sort_keys=True)}"


def _rule_reason(rule: dict) -> str:
    """Short human label for a rule — used in signal labels (the 'reasons')."""
    rid = rule["id"]
    p = rule.get("params") or {}

    def fmt(v):
        return f"{float(v):g}"

    if rid == "ema_cross_up":
        return f"EMA {_int(p,'fast',9)} ↑ EMA {_int(p,'slow',21)}"
    if rid == "ema_cross_down":
        return f"EMA {_int(p,'fast',9)} ↓ EMA {_int(p,'slow',21)}"
    if rid == "sma_cross_up":
        return f"SMA {_int(p,'fast',10)} ↑ SMA {_int(p,'slow',30)}"
    if rid == "sma_cross_down":
        return f"SMA {_int(p,'fast',10)} ↓ SMA {_int(p,'slow',30)}"
    if rid == "macd_cross_up":
        return "MACD ↑ signal"
    if rid == "macd_cross_down":
        return "MACD ↓ signal"
    if rid == "rsi_cross_up":
        return f"RSI {_int(p,'length',14)} ↑ {fmt(_float(p,'level',50))}"
    if rid == "rsi_cross_down":
        return f"RSI {_int(p,'length',14)} ↓ {fmt(_float(p,'level',50))}"
    if rid == "bb_lower_touch":
        return f"Low ≤ BB low ({_int(p,'length',20)},{fmt(_float(p,'mult',2.0))})"
    if rid == "bb_upper_touch":
        return f"High ≥ BB up ({_int(p,'length',20)},{fmt(_float(p,'mult',2.0))})"
    if rid == "supertrend_up":
        return "Supertrend ↑"
    if rid == "supertrend_down":
        return "Supertrend ↓"
    if rid == "breakout_high":
        return f"Breakout {_int(p,'lookback',20)} high"
    if rid == "breakdown_low":
        return f"Breakdown {_int(p,'lookback',20)} low"
    if rid == "support_bounce":
        return "Support bounce"
    if rid == "resistance_reject":
        return "Resistance reject"
    if rid == "ichimoku_cloud_up":
        return "Close ↑ cloud"
    if rid == "ichimoku_cloud_down":
        return "Close ↓ cloud"
    if rid == "ichimoku_above_cloud":
        return "Above cloud"
    if rid == "ichimoku_below_cloud":
        return "Below cloud"
    if rid == "ema_stack_bull":
        return f"EMA stack ↑ ({_int(p,'fast',10)}>{_int(p,'mid',20)}>{_int(p,'slow',50)})"
    if rid == "ema_stack_bear":
        return f"EMA stack ↓ ({_int(p,'fast',10)}<{_int(p,'mid',20)}<{_int(p,'slow',50)})"
    if rid == "rsi_below":
        return f"RSI {_int(p,'length',14)} < {fmt(_float(p,'level',50))}"
    if rid == "rsi_above":
        return f"RSI {_int(p,'length',14)} > {fmt(_float(p,'level',50))}"
    if rid == "price_above_ema":
        return f"Price > EMA {_int(p,'length',200)}"
    if rid == "price_below_ema":
        return f"Price < EMA {_int(p,'length',200)}"
    if rid == "vwap_above":
        return "Above VWAP"
    if rid == "vwap_below":
        return "Below VWAP"
    if rid == "adx_above":
        return f"ADX > {fmt(_float(p,'level',25))}"
    if rid == "volume_spike":
        return f"Volume ×{fmt(_float(p,'mult',2.0))} avg"
    if rid == "session":
        return "In session"
    if rid == "halftrend_up":
        return "HalfTrend ↑ (bull flip)"
    if rid == "halftrend_down":
        return "HalfTrend ↓ (bear flip)"
    if rid == "halftrend_bull":
        return "HalfTrend bull"
    if rid == "halftrend_bear":
        return "HalfTrend bear"
    return rid


def _reason_join(*rule_lists) -> str:
    labels = [_rule_reason(r) for rules in rule_lists for r in rules]
    return " · ".join(labels)


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
    credit = ""
    if any(
        r["id"].startswith("halftrend")
        for bucket in ("entry", "exit", "short_entry", "short_exit", "filters", "short_filters")
        for r in strategy.get("rules", {}).get(bucket, [])
    ):
        credit = (
            f"//  HalfTrend logic: everget — adapted from BigBeluga's HalfTrend "
            f"Signal Engine (CC BY-NC-SA 4.0)\n"
        )
    return (
        f"// ════════════════════════════════════════════════════════════════\n"
        f"//  {_safe_name(strategy['name'])}  (revision {strategy.get('revision', 1)})\n"
        f"//  Generated by the Trading Brain — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"//  Source: {src}\n"
        f"//  Market: {strategy.get('market', 'any')} · timeframe {strategy.get('timeframe', 'any')} · side {strategy.get('side', 'long')}\n"
        f"//  {rules}\n"
        f"//  NOTE: educational tool — not financial advice. Test before live use.\n"
        f"{credit}"
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
    """Compile the strategy into a Pine v6 ``indicator()`` script.

    The generated script is organised in neat, commented blocks:

      INPUTS · INDICATOR COMPUTATIONS · TRADE STATE · LONG BLOCK ·
      SHORT BLOCK · POSITION ENGINE · VISUALS · INDICATOR LINES ·
      POSITION PANEL · ALERTS

    Every entry/exit signal draws an on-chart label showing the exact
    **reasons** it fired (which rules were true), and a small live table
    (top-right) shows the current position, entry/stop/target and reason.
    """
    strategy = normalize_strategy(strategy)
    side = strategy.get("side", "long")
    do_long = side in ("long", "both")
    do_short = side in ("short", "both")
    name = _safe_name(strategy["name"])

    ctx = _Ctx()
    buy_msg, sell_msg, short_msg, cover_msg = _alert_messages(strategy)

    # ── python-side compilation of conditions + reason texts ──────────
    long_cond = long_exit = "false"
    short_cond = short_exit = "false"
    long_reason = long_exit_msg = "—"
    short_reason = short_exit_msg = "—"
    if do_long:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "entry", "exit", "filters")
        long_cond = _join([e for e in (entry, filters) if e != "false"])
        long_exit = exit_
        long_reason = _reason_join(strategy["rules"]["entry"], strategy["rules"]["filters"]) or "—"
        long_exit_msg = _reason_join(strategy["rules"]["exit"]) or "—"
    if do_short:
        entry, exit_, filters = _build_side(ctx, strategy["rules"], "short_entry", "short_exit", "short_filters")
        short_cond = _join([e for e in (entry, filters) if e != "false"])
        short_exit = exit_
        short_reason = _reason_join(strategy["rules"]["short_entry"], strategy["rules"]["short_filters"]) or "—"
        short_exit_msg = _reason_join(strategy["rules"]["short_exit"]) or "—"

    stop = strategy.get("stop")
    target = strategy.get("target")
    has_stop = stop is not None
    has_target = target is not None and target.get("type") != "opposite"

    long_stop_assigns = _stop_plan(ctx, strategy, "longStop", "long")
    short_stop_assigns = _stop_plan(ctx, strategy, "shortStop", "short")
    long_target_assign = _target_assign(strategy, "longTarget", "longEntry", "longStop", "long")
    short_target_assign = _target_assign(strategy, "shortTarget", "shortEntry", "shortStop", "short")

    # exit-why booleans (each labelled in the signal reasons)
    long_hits, short_hits = [], []
    if do_long:
        long_hits.append("longStopHit = inLong and not na(longStop) and low <= longStop" if has_stop else "longStopHit = false")
        long_hits.append("longTargetHit = inLong and not na(longTarget) and high >= longTarget" if has_target else "longTargetHit = false")
        long_hits.append(
            "longMaxBars = bar_index - longBar >= maxBarsInTrade"
            if long_exit == "false" and not has_stop and not has_target
            else "longMaxBars = false"
        )
    if do_short:
        short_hits.append("shortStopHit = inShort and not na(shortStop) and high >= shortStop" if has_stop else "shortStopHit = false")
        short_hits.append("shortTargetHit = inShort and not na(shortTarget) and low <= shortTarget" if has_target else "shortTargetHit = false")
        short_hits.append(
            "shortMaxBars = bar_index - shortBar >= maxBarsInTrade"
            if short_exit == "false" and not has_stop and not has_target
            else "shortMaxBars = false"
        )

    L = []
    L.append("//@version=6")
    L.append(f'indicator("{name} [Brain]", shorttitle="Brain {_slug(name)}", overlay=true, max_labels_count=500)')
    L.append("")
    L.append(_header(strategy))
    L.append("")
    L.append("// ┌─ 1 · INPUTS ─────────────────────────────────────────────────────")
    L.append("maxBarsInTrade = input.int(100, 'Max bars in trade', minval=5)")
    L.append("")
    L.append("// ┌─ 2 · INDICATOR COMPUTATIONS ─────────────────────────────────────")
    L.extend(ctx.decls or ["// (no indicator computations)"])
    L.append("")
    L.append("// ┌─ 3 · TRADE STATE ────────────────────────────────────────────────")
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
    L.append('var string posReason = "—"   // why the current position is open')
    L.append("")
    if do_long:
        L.append("// ┌─ 4 · LONG BLOCK ────── conditions · reasons · engine ─────────────")
        L.append(f"longCond      = {long_cond}")
        L.append(f"longExitCond  = {long_exit}")
        L.append(f'longReason    = "{long_reason}"   // shown on every BUY label')
        L.append(f'longExitMsg   = "{long_exit_msg}"')
        L.extend(long_hits)
        L.append(
            "longExitReason = longExitCond ? longExitMsg : longStopHit ? \"Stop hit\" "
            ": longTargetHit ? \"Target hit\" : \"Max bars in trade\""
        )
        L.append("longTrigger     = longCond and not inLong")
        L.append("longExitTrigger = inLong and (longExitCond or longStopHit or longTargetHit or longMaxBars)")
        L.append("")
    if do_short:
        L.append("// ┌─ 5 · SHORT BLOCK ───── conditions · reasons · engine ─────────────")
        L.append(f"shortCond      = {short_cond}")
        L.append(f"shortExitCond  = {short_exit}")
        L.append(f'shortReason    = "{short_reason}"   // shown on every SHORT label')
        L.append(f'shortExitMsg   = "{short_exit_msg}"')
        L.extend(short_hits)
        L.append(
            "shortExitReason = shortExitCond ? shortExitMsg : shortStopHit ? \"Stop hit\" "
            ": shortTargetHit ? \"Target hit\" : \"Max bars in trade\""
        )
        L.append("shortTrigger     = shortCond and not inShort")
        L.append("shortExitTrigger = inShort and (shortExitCond or shortStopHit or shortTargetHit or shortMaxBars)")
        L.append("")
    L.append("// ┌─ 6 · POSITION ENGINE ── entries · exits · reason labels · alerts ──")
    if do_long and do_short:
        L += [
            "// flip: long signal while short → cover first, then enter long",
            "if longTrigger and inShort",
            "    inShort := false",
            "    shortEntry := na",
            "    shortBar := na",
            "    shortStop := na",
            "    shortTarget := na",
            '    posReason := "—"',
            f"    alert({cover_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "// flip: short signal while long → sell first, then enter short",
            "if shortTrigger and inLong",
            "    inLong := false",
            "    longEntry := na",
            "    longBar := na",
            "    longStop := na",
            "    longTarget := na",
            '    posReason := "—"',
            f"    alert({sell_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    if do_long:
        L += [
            "// — long entry",
            "if longTrigger",
            "    inLong := true",
            "    longEntry := close",
            "    longBar := bar_index",
            *[f"    {a}" for a in long_stop_assigns],
            f"    {long_target_assign}",
            "    posReason := longReason",
            f'    label.new(bar_index, low, "▲ BUY\\n" + longReason, style=label.style_label_up, color=color.new(#26a69a, 100), textcolor=color.white, size=size.small, yloc=yloc.belowbar)',
            f"    alert({buy_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "// — long exit",
            "if longExitTrigger",
            "    inLong := false",
            "    longEntry := na",
            "    longBar := na",
            "    longStop := na",
            "    longTarget := na",
            '    posReason := "—"',
            f'    label.new(bar_index, high, "▼ SELL\\n" + longExitReason, style=label.style_label_down, color=color.new(#ef5350, 100), textcolor=color.white, size=size.small, yloc=yloc.abovebar)',
            f"    alert({sell_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    if do_short:
        L += [
            "// — short entry",
            "if shortTrigger",
            "    inShort := true",
            "    shortEntry := close",
            "    shortBar := bar_index",
            *[f"    {a}" for a in short_stop_assigns],
            f"    {short_target_assign}",
            "    posReason := shortReason",
            f'    label.new(bar_index, high, "▼ SHORT\\n" + shortReason, style=label.style_label_down, color=color.new(#ff7043, 100), textcolor=color.white, size=size.small, yloc=yloc.abovebar)',
            f"    alert({short_msg!r}, alert.freq_once_per_bar_close)",
            "",
            "// — short exit (cover)",
            "if shortExitTrigger",
            "    inShort := false",
            "    shortEntry := na",
            "    shortBar := na",
            "    shortStop := na",
            "    shortTarget := na",
            '    posReason := "—"',
            f'    label.new(bar_index, low, "▲ COVER\\n" + shortExitReason, style=label.style_label_up, color=color.new(#42a5f5, 100), textcolor=color.white, size=size.small, yloc=yloc.belowbar)',
            f"    alert({cover_msg!r}, alert.freq_once_per_bar_close)",
            "",
        ]
    L.append("// ┌─ 7 · VISUALS ────────── arrows · background · stop/target levels ───")
    if do_long:
        L.append("plotshape(longTrigger, 'BUY', style=shape.triangleup, location=location.belowbar, color=color.new(#26a69a, 0), size=size.tiny)")
        L.append("plotshape(longExitTrigger, 'SELL', style=shape.triangledown, location=location.abovebar, color=color.new(#ef5350, 0), size=size.tiny)")
    if do_short:
        L.append("plotshape(shortTrigger, 'SHORT', style=shape.triangledown, location=location.abovebar, color=color.new(#ff7043, 0), size=size.tiny)")
        L.append("plotshape(shortExitTrigger, 'COVER', style=shape.triangleup, location=location.belowbar, color=color.new(#42a5f5, 0), size=size.tiny)")
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
    if ctx.plots:
        L.append("// ┌─ 8 · INDICATOR LINES ──────────────────────────────────────────────")
        L.extend(ctx.plots)
        L.append("")
    # live position-panel cell expressions (python-built, per side config)
    if do_long and do_short:
        pos_expr = 'inLong ? "LONG ▲" : inShort ? "SHORT ▼" : "FLAT —"'
        pos_color = "inLong ? color.new(#26a69a, 0) : inShort ? color.new(#ef5350, 0) : color.new(#787b86, 0)"
        entry_expr = 'inLong ? str.tostring(longEntry) : inShort ? str.tostring(shortEntry) : "—"'
        stop_expr = 'inLong ? str.tostring(longStop) : inShort ? str.tostring(shortStop) : "—"'
        target_expr = 'inLong ? str.tostring(longTarget) : inShort ? str.tostring(shortTarget) : "—"'
    elif do_long:
        pos_expr = 'inLong ? "LONG ▲" : "FLAT —"'
        pos_color = "inLong ? color.new(#26a69a, 0) : color.new(#787b86, 0)"
        entry_expr = 'inLong ? str.tostring(longEntry) : "—"'
        stop_expr = 'inLong ? str.tostring(longStop) : "—"'
        target_expr = 'inLong ? str.tostring(longTarget) : "—"'
    else:
        pos_expr = 'inShort ? "SHORT ▼" : "FLAT —"'
        pos_color = "inShort ? color.new(#ef5350, 0) : color.new(#787b86, 0)"
        entry_expr = 'inShort ? str.tostring(shortEntry) : "—"'
        stop_expr = 'inShort ? str.tostring(shortStop) : "—"'
        target_expr = 'inShort ? str.tostring(shortTarget) : "—"'

    L.append("// ┌─ 9 · POSITION PANEL ── live table (top-right) ─────────────────────")
    L.append("var table posTable = table.new(position.top_right, 2, 6, bgcolor=color.new(#1e222d, 90), border_width=1, border_color=color.new(#363a45, 100))")
    L.append("if barstate.isfirst")
    L.append("    table.merge_cells(posTable, 0, 0, 1, 0)")
    L.append("if barstate.islast")
    L.append(f'    table.cell(posTable, 0, 0, "TRADING BRAIN · {name}", text_color=color.new(#2962ff, 0), text_size=size.small)')
    L.append('    table.cell(posTable, 0, 1, "Position", text_color=color.new(#787b86, 0), text_size=size.small)')
    L.append(f"    table.cell(posTable, 1, 1, {pos_expr}, text_color={pos_color}, text_size=size.small)")
    L.append('    table.cell(posTable, 0, 2, "Entry", text_color=color.new(#787b86, 0), text_size=size.small)')
    L.append(f"    table.cell(posTable, 1, 2, {entry_expr}, text_size=size.small)")
    L.append('    table.cell(posTable, 0, 3, "Stop", text_color=color.new(#787b86, 0), text_size=size.small)')
    L.append(f"    table.cell(posTable, 1, 3, {stop_expr}, text_color=color.new(#ef5350, 0), text_size=size.small)")
    L.append('    table.cell(posTable, 0, 4, "Target", text_color=color.new(#787b86, 0), text_size=size.small)')
    L.append(f"    table.cell(posTable, 1, 4, {target_expr}, text_color=color.new(#26a69a, 0), text_size=size.small)")
    L.append('    table.cell(posTable, 0, 5, "Reason", text_color=color.new(#787b86, 0), text_size=size.small)')
    L.append("    table.cell(posTable, 1, 5, posReason, text_size=size.small)")
    L.append("")
    L.append("// ┌─ 10 · ALERTS ──────── wire these to your brain webhook ─────────────")
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
