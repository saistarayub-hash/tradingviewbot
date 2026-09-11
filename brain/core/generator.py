"""Strategy → Pine Script (v6) generator.

The brain never free-styles Pine code: it extracts a *strategy* (a small
JSON object of known rules) and this module compiles it into deterministic,
syntactically-safe Pine Script v6. Two artefacts are produced:

  * indicator           — plots BUY/SELL signals, tracks stops/targets and
                          fires ``alertcondition`` webhooks at the brain
                          (the live loop the brain learns from).
  * backtest strategy   — the same rules as a ``strategy()`` script so the
                          logic can be backtested on TradingView.

Rule ids understood by the brain (tell the LLM / heuristics only these exist):

  entry   : ema_cross_up, sma_cross_up, macd_cross_up, rsi_cross_up,
            bb_lower_touch, support_bounce, breakout_high, supertrend_up
  exit    : ema_cross_down, sma_cross_down, macd_cross_down, rsi_cross_down,
            bb_upper_touch, supertrend_down, resistance_reject
  filters : rsi_below, rsi_above, price_above_ema, price_below_ema,
            vwap_above, vwap_below, adx_above, volume_spike, session

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
    "side": "long",
    "summary": "",
    "rules": {"entry": [], "exit": [], "filters": []},
    "stop": None,      # {"type": "atr"|"percent"|"swing", "params": {...}}
    "target": None,    # {"type": "rr"|"percent"|"opposite", "params": {...}}
    "notes": [],
    "revision": 1,
    "updated_at": None,
}

KNOWN_RULE_IDS = {
    "ema_cross_up", "sma_cross_up", "macd_cross_up", "rsi_cross_up",
    "bb_lower_touch", "support_bounce", "breakout_high", "supertrend_up",
    "ema_cross_down", "sma_cross_down", "macd_cross_down", "rsi_cross_down",
    "bb_upper_touch", "supertrend_down", "resistance_reject",
    "rsi_below", "rsi_above", "price_above_ema", "price_below_ema",
    "vwap_above", "vwap_below", "adx_above", "volume_spike", "session",
}

DEFAULT_STOP = {"type": "atr", "params": {"length": 14, "mult": 2.0}}
DEFAULT_TARGET = {"type": "rr", "params": {"rr": 2.0}}


def default_strategy() -> dict:
    """Seed strategy shipped with the repo (EMA 9/21 + RSI filter)."""
    s = dict(CANONICAL)
    s.update(
        name="EMA Crossover",
        timeframe="15",
        market="crypto",
        summary=(
            "Long when the 9 EMA crosses above the 21 EMA while RSI(14) is below 50; "
            "exit when the 9 EMA crosses back below. 2×ATR stop, 2R target."
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
            for bucket in ("entry", "exit", "filters"):
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
    if not s["rules"]["entry"]:
        s["rules"]["entry"] = [{"id": "ema_cross_up", "params": {"fast": 9, "slow": 21}}]
    if name:
        s["name"] = name
    if source_video:
        s["source_video"] = source_video
    s["side"] = "long"
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
}
_EXIT_IDS = {
    "ema_cross_down", "sma_cross_down", "macd_cross_down", "rsi_cross_down",
    "bb_upper_touch", "supertrend_down", "resistance_reject",
}
_FILTER_IDS = {
    "rsi_below", "rsi_above", "price_above_ema", "price_below_ema",
    "vwap_above", "vwap_below", "adx_above", "volume_spike", "session",
}


def apply_patch(strategy: dict, patch: dict) -> dict:
    """Merge a patch (e.g. from the learning loop) into the strategy."""
    s = json.loads(json.dumps(strategy))
    for bucket in ("entry", "exit", "filters"):
        add = (patch.get("rules") or {}).get(bucket)
        if isinstance(add, list):
            s["rules"][bucket] = _clean_rules(s["rules"].get(bucket, []) + add)
    if isinstance(patch.get("rules"), dict) and isinstance(patch["rules"].get("remove"), list):
        for rid in patch["rules"]["remove"]:
            for bucket in ("entry", "exit", "filters"):
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
        self._series: dict[str, str] = {}   # expression → var name
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
    expr = f"ta.macd(close, {fast}, {slow}, {smooth})"
    if expr in ctx._series:
        m, sig, hist = ctx._series[expr]
    else:
        ctx._n += 1
        m, sig, hist = f"macdLine{ctx._n}", f"macdSignal{ctx._n}", f"macdHist{ctx._n}"
        ctx._series[expr] = (m, sig, hist)
        ctx.decls.append(f"[{m}, {sig}, {hist}] = {expr}")
        ctx.plot(hist, "MACD hist", color="color.new(#888888, 0)", style="plot.style_columns")
    return f"ta.crossover({m}, {sig})" if up else f"ta.crossunder({m}, {sig})"


def _rule_rsi_cross(ctx, p, up: bool) -> str:
    length, level = _int(p, "length", 14), _float(p, "level", 50)
    r = ctx.series(f"ta.rsi(close, {length})", "rsi")
    return f"ta.crossover({r}, {level})" if up else f"ta.crossunder({r}, {level})"


def _rule_bb_touch(ctx, p, lower: bool) -> str:
    length, mult = _int(p, "length", 20), _float(p, "mult", 2.0)
    expr = f"ta.bb(close, {length}, {mult})"
    if expr in ctx._series:
        b, bUp, bLow = ctx._series[expr]
    else:
        ctx._n += 1
        b, bUp, bLow = f"bbMid{ctx._n}", f"bbUp{ctx._n}", f"bbLow{ctx._n}"
        ctx._series[expr] = (b, bUp, bLow)
        ctx.decls.append(f"[{b}, {bUp}, {bLow}] = {expr}")
        ctx.plot(bUp, "BB upper", color="color.new(#aaaaaa, 60)")
        ctx.plot(bLow, "BB lower", color="color.new(#aaaaaa, 60)")
    return f"low <= {bLow}" if lower else f"high >= {bUp}"


def _rule_supertrend(ctx, p, up: bool) -> str:
    factor, atr_period = _float(p, "factor", 3.0), _int(p, "atr", 10)
    expr = f"ta.supertrend({factor}, {atr_period})"
    if expr in ctx._series:
        st, dr = ctx._series[expr]
    else:
        ctx._n += 1
        st, dr = f"stTrend{ctx._n}", f"stDir{ctx._n}"
        ctx._series[expr] = (st, dr)
        ctx.decls.append(f"[{st}, {dr}] = {expr}")
        ctx.plot(
            st, "Supertrend",
            color=f"{dr} < 0 ? color.new(#26a69a, 0) : color.new(#ef5350, 0)",
        )
    return f"{dr} < 0" if up else f"{dr} > 0"


def _rule_breakout(ctx, p) -> str:
    lookback = _int(p, "lookback", 20)
    return f"high > ta.highest(high, {lookback})[1]"


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
    expr = f"ta.dmi({length}, {length})"
    if expr in ctx._series:
        d, m, a = ctx._series[expr]
    else:
        ctx._n += 1
        d, m, a = f"diPlus{ctx._n}", f"diMinus{ctx._n}", f"adx{ctx._n}"
        ctx._series[expr] = (d, m, a)
        ctx.decls.append(f"[{d}, {m}, {a}] = {expr}")
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
    "support_bounce": _rule_support_bounce,
    "resistance_reject": _rule_resistance_reject,
    "rsi_below": lambda c, p: _rule_rsi_filter(c, p, True),
    "rsi_above": lambda c, p: _rule_rsi_filter(c, p, False),
    "price_above_ema": lambda c, p: _rule_price_ema(c, p, True),
    "price_below_ema": lambda c, p: _rule_price_ema(c, p, False),
    "vwap_above": lambda c, p: _rule_vwap(c, p, True),
    "vwap_below": lambda c, p: _rule_vwap(c, p, False),
    "adx_above": _rule_adx,
    "volume_spike": _rule_volume,
    "session": _rule_session,
}


def _rule_label(rule: dict) -> str:
    p = rule.get("params") or {}
    return f"{rule['id']} {json.dumps(p, sort_keys=True)}"


def summarize_rules(strategy: dict) -> str:
    """Human-readable strategy summary for chat replies and code headers."""
    lines = []
    for bucket, label in (("entry", "Entry"), ("exit", "Exit"), ("filters", "Filters")):
        rules = strategy["rules"].get(bucket, [])
        text = ", ".join(_rule_label(r) for r in rules) if rules else "—"
        lines.append(f"{label}: {text}")
    stop = strategy.get("stop")
    target = strategy.get("target")
    if stop:
        lines.append(f"Stop: {stop['type']} {json.dumps(stop.get('params') or {}, sort_keys=True)}")
    if target:
        lines.append(f"Target: {target['type']} {json.dumps(target.get('params') or {}, sort_keys=True)}")
    return "\n".join(lines)


def _build_conditions(strategy: dict):
    """Compile rule lists into Pine condition expressions."""
    ctx = _Ctx()
    entry = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in strategy["rules"]["entry"]])
    exit_ = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in strategy["rules"]["exit"]])
    filters = _join([_RULES[r["id"]](ctx, r.get("params") or {}) for r in strategy["rules"]["filters"]])
    return ctx, entry, exit_, filters


def _join(exprs) -> str:
    exprs = [e for e in exprs if e]
    if not exprs:
        return "false"
    if len(exprs) == 1:
        return exprs[0]
    return "(" + " and ".join(exprs) + ")"


def _stop_plan(strategy: dict) -> tuple[list[str], list[str]]:
    """Return ``(hoisted_decls, assignment_lines)`` for the stop.

    Hoisting matters: Pine does not allow local declarations inside ``if``
    blocks, so any helper series (e.g. a swing pivot) must be top-level.
    """
    stop = strategy.get("stop")
    if not stop:
        return [], ["stopLevel := na"]
    p = stop.get("params") or {}
    if stop["type"] == "atr":
        return [], [f"stopLevel := close - ta.atr({_int(p, 'length', 14)}) * {_float(p, 'mult', 2.0)}"]
    if stop["type"] == "percent":
        return [], [f"stopLevel := close * (1 - {_float(p, 'pct', 1.0)} / 100.0)"]
    if stop["type"] == "swing":
        left, right = _int(p, "left", 10), _int(p, "right", 10)
        return (
            [f"swingStop = ta.pivotlow(low, {left}, {right})"],
            ["stopLevel := na(swingStop) ? close * 0.99 : swingStop"],
        )
    return [], ["stopLevel := na"]


def _target_assignment(strategy: dict) -> str:
    target = strategy.get("target")
    if not target:
        return "targetLevel := na"
    p = target.get("params") or {}
    if target["type"] == "rr":
        return f"targetLevel := entryPrice + (entryPrice - stopLevel) * {_float(p, 'rr', 2.0)}"
    if target["type"] == "percent":
        return f"targetLevel := entryPrice * (1 + {_float(p, 'pct', 2.0)} / 100.0)"
    return "targetLevel := na"


def _header(strategy: dict) -> str:
    src = strategy.get("source_video") or "hand-authored"
    rules = summarize_rules(strategy).replace("\n", "\n//  ")
    return (
        f"// ════════════════════════════════════════════════════════════════\n"
        f"//  {strategy['name']}  (revision {strategy.get('revision', 1)})\n"
        f"//  Generated by the Trading Brain — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"//  Source: {src}\n"
        f"//  Market: {strategy.get('market', 'any')} · timeframe {strategy.get('timeframe', 'any')} · side {strategy.get('side', 'long')}\n"
        f"//  {rules}\n"
        f"//  NOTE: educational tool — not financial advice. Test before live use.\n"
        f"// ════════════════════════════════════════════════════════════════"
    )


def _alert_messages(strategy: dict) -> tuple[str, str]:
    base = (
        f'{{"strategy":"{strategy["name"]}",'
        f'"revision":{strategy.get("revision", 1)},'
        f'"action":"BUY","symbol":"{{{{ticker}}}}","interval":"{{{{interval}}}}",'
        f'"price":{{{{close}}}},"time":"{{{{time}}}}"}}'
    )
    sell = base.replace('"BUY"', '"SELL"')
    return base, sell


def generate_indicator(strategy: dict) -> str:
    """Compile the strategy into a Pine v6 ``indicator()`` script with alerts."""
    strategy = normalize_strategy(strategy)
    ctx, entry, exit_, filters = _build_conditions(strategy)
    buy_msg, sell_msg = _alert_messages(strategy)

    long_cond = _join([e for e in [entry, filters] if e != "false"])
    exit_cond = exit_

    stop = strategy.get("stop")
    target = strategy.get("target")
    has_stop = stop is not None
    has_target = target is not None and target.get("type") != "opposite"
    stop_hoisted, stop_assigns = _stop_plan(strategy)
    for d in stop_hoisted:
        ctx.decls.append(d)

    stop_guard = ""
    target_guard = ""
    if has_stop:
        stop_guard = " or (inLong and not na(stopLevel) and low <= stopLevel)"
    if has_target:
        target_guard = " or (inLong and not na(targetLevel) and high >= targetLevel)"

    decls = "\n".join(f"{d}" for d in ctx.decls) or "// (no indicator declarations)"
    plots = "\n".join(ctx.plots)

    exit_fallback = ""
    if exit_cond == "false" and not has_stop and not has_target:
        exit_fallback = " or bar_index - entryBar >= maxBarsInTrade"

    stop_block = "\n".join(f"    {line}" for line in stop_assigns)

    lines = []
    lines.append("//@version=6")
    lines.append(f'indicator("{strategy["name"]} [Brain]", shorttitle="Brain {_slug(strategy["name"])}", overlay=true, max_labels_count=500)')
    lines.append("")
    lines.append(_header(strategy))
    lines.append("")
    lines.append("// ── Inputs ─────────────────────────────────────────────────────────")
    lines.append("maxBarsInTrade = input.int(100, 'Max bars in trade', minval=5)")
    lines.append("")
    lines.append("// ── Indicator computations ────────────────────────────────────────")
    lines.append(decls)
    lines.append("")
    lines.append("// ── Trade state ───────────────────────────────────────────────────")
    lines.append("var bool  inLong     = false")
    lines.append("var float entryPrice = na")
    lines.append("var float stopLevel  = na")
    lines.append("var float targetLevel = na")
    lines.append("var int   entryBar   = na")
    lines.append("")
    lines.append("// ── Conditions ────────────────────────────────────────────────────")
    lines.append(f"longCond  = {long_cond}")
    lines.append(f"exitCond  = {exit_cond}")
    lines.append("longTrigger = longCond and not inLong")
    lines.append(f"exitTrigger = inLong and (exitCond{stop_guard}{target_guard}{exit_fallback})")
    lines.append("")
    lines.append("// ── Position engine ───────────────────────────────────────────────")
    lines.append("if longTrigger")
    lines.append("    inLong := true")
    lines.append("    entryPrice := close")
    lines.append("    entryBar := bar_index")
    lines.append(stop_block)
    lines.append(f"    {_target_assignment(strategy)}")
    lines.append(f'    alert({buy_msg!r}, alert.freq_once_per_bar_close)')
    lines.append("")
    lines.append("if exitTrigger")
    lines.append("    inLong := false")
    lines.append("    entryPrice := na")
    lines.append("    entryBar := na")
    lines.append("    stopLevel := na")
    lines.append("    targetLevel := na")
    lines.append(f'    alert({sell_msg!r}, alert.freq_once_per_bar_close)')
    lines.append("")
    lines.append("// ── Visuals ───────────────────────────────────────────────────────")
    lines.append("plotshape(longTrigger, 'BUY', style=shape.triangleup, location=location.belowbar, color=color.new(#26a69a, 0), size=size.small)")
    lines.append("plotshape(exitTrigger, 'SELL', style=shape.triangledown, location=location.abovebar, color=color.new(#ef5350, 0), size=size.small)")
    lines.append("bgcolor(inLong ? color.new(#26a69a, 92) : na)")
    if has_stop:
        lines.append("plot(inLong ? stopLevel : na, 'Stop', style=plot.style_linebr, color=color.new(#ef5350, 20), linewidth=2)")
    if has_target:
        lines.append("plot(inLong ? targetLevel : na, 'Target', style=plot.style_linebr, color=color.new(#26a69a, 20), linewidth=2)")
    lines.append("")
    if plots:
        lines.append("// ── Indicator lines ───────────────────────────────────────────────")
        lines.append(plots)
        lines.append("")
    lines.append("// ── Alerts → brain webhook ────────────────────────────────────────")
    lines.append(f'alertcondition(longCond, "Brain BUY — {strategy["name"]}", message={buy_msg!r})')
    lines.append(f'alertcondition(exitCond, "Brain SELL — {strategy["name"]}", message={sell_msg!r})')
    return "\n".join(lines) + "\n"


def generate_strategy(strategy: dict) -> str:
    """Compile the strategy into a Pine v6 ``strategy()`` backtest script."""
    strategy = normalize_strategy(strategy)
    ctx, entry, exit_, filters = _build_conditions(strategy)
    long_cond = _join([e for e in [entry, filters] if e != "false"])

    stop = strategy.get("stop")
    target = strategy.get("target")
    has_stop = stop is not None
    has_target = target is not None and target.get("type") != "opposite"
    stop_hoisted, stop_assigns = _stop_plan(strategy)
    for d in stop_hoisted:
        ctx.decls.append(d)

    lines = []
    lines.append("//@version=6")
    lines.append(
        f'strategy("{strategy["name"]} [Brain BT]", shorttitle="Brain {_slug(strategy["name"])} BT", '
        f'overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, '
        f'default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.1, pyramiding=0)'
    )
    lines.append("")
    lines.append(_header(strategy))
    lines.append("")
    lines.append("// ── Indicator computations ────────────────────────────────────────")
    for d in ctx.decls:
        lines.append(d)
    lines.append("")
    lines.append("// ── Conditions ────────────────────────────────────────────────────")
    lines.append(f"longCond  = {long_cond}")
    lines.append(f"exitCond  = {exit_}")
    lines.append("")
    lines.append("// ── Position state (stop / target captured at entry) ─────────────")
    lines.append("var float entryPrice = na")
    lines.append("var float stopLevel = na")
    lines.append("var float targetLevel = na")
    lines.append("")
    lines.append("if longCond and strategy.position_size == 0")
    lines.append("    strategy.entry('Long', strategy.long)")
    lines.append("    entryPrice := close")
    lines.append("\n".join(f"    {line}" for line in stop_assigns))
    lines.append(f"    {_target_assignment(strategy)}")
    lines.append("")
    lines.append("if exitCond and strategy.position_size > 0")
    lines.append("    strategy.close('Long', comment='Signal exit')")
    lines.append("")
    lines.append("// ── Risk management exit ──────────────────────────────────────────")
    if has_stop or has_target:
        parts = []
        guards = []
        if has_stop:
            parts.append("stop=stopLevel")
            guards.append("not na(stopLevel)")
        if has_target:
            parts.append("limit=targetLevel")
            guards.append("not na(targetLevel)")
        lines.append(f"if strategy.position_size > 0 and {' and '.join(guards)}")
        lines.append(f"    strategy.exit('Risk', from_entry='Long', {', '.join(parts)})")
    else:
        lines.append("// (no stop/target configured)")
    lines.append("")
    if ctx.plots:
        lines.append("// ── Indicator lines ───────────────────────────────────────────────")
        for p in ctx.plots:
            lines.append(p)
    return "\n".join(lines) + "\n"


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", name)[:20]
    return slug or "Strategy"
