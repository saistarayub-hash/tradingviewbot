"""The learning loop.

TradingView indicator → alert webhook → this module.

Every BUY/SELL the generated indicator fires lands here. Pairs are matched
by symbol+timeframe, and when a SELL arrives the brain measures the outcome
and (once enough samples exist) proposes an adjustment:

  * win rate low   → tighten risk (raise R:R target, or reduce ATR mult)
  * win rate high  → relax risk (let winners run with a bigger target)
  * ratio too low  → suggest lowering it so more trades reach the target

Suggestions are stored and surfaced in chat/status — the user approves
changes (the brain never silently rewrites live trading logic).
"""
from __future__ import annotations

MIN_SAMPLES = 5


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_stop_hit(signal: dict) -> bool:
    """Heuristic: exit below the recorded stop level → stopped out."""
    stop = _num(signal.get("stopLevel"))
    price = _num(signal.get("price"))
    return stop is not None and price is not None and price < stop


def record(stores, payload: dict) -> dict:
    """Store one webhook payload and update learning stats."""
    signals = stores.signals.read()
    data = dict(payload)
    data["ts"] = payload.get("time") or None
    stores.signals.write(signals + [data])
    return data


def review(stores, strategy: dict | None) -> dict:
    """Compute stats + a suggestion from the signal history."""
    signals = stores.signals.read()
    buys = [s for s in signals if s.get("action") == "BUY"]
    sells = [s for s in signals if s.get("action") == "SELL"]
    pairs = _pair_up(buys, sells)

    stats = {
        "signals": len(signals),
        "buys": len(buys),
        "sells": len(sells),
        "completed": len(pairs),
        "wins": 0,
        "losses": 0,
        "winRate": None,
        "avgRR": None,
        "ready": len(pairs) >= MIN_SAMPLES,
    }

    wins = 0.0
    total_pnl = 0.0
    trades = []
    for b, s in pairs:
        bp, sp = _num(b.get("price")), _num(s.get("price"))
        if bp is None or sp is None:
            continue
        won = sp >= bp
        wins += 1.0 if won else 0.0
        total_pnl += (sp - bp) / bp
        trades.append({"entry": bp, "exit": sp, "win": won})
    n = len(trades)
    if n:
        stats["wins"] = int(wins)
        stats["losses"] = n - int(wins)
        stats["winRate"] = round(wins / n, 3)
        stats["avgRR"] = round(total_pnl / n, 4)

    suggestion = _suggest(stats, strategy)
    if suggestion:
        stats["suggestion"] = suggestion
    return stats


def _pair_up(buys, sells):
    """FIFO pairing of BUY→SELL per symbol+timeframe."""
    open_positions: dict[tuple, list[dict]] = {}
    pairs = []
    for sig in sorted(buys + sells, key=lambda s: s.get("ts") or ""):
        key = (sig.get("symbol"), sig.get("interval"))
        bucket = open_positions.setdefault(key, [])
        if sig.get("action") == "BUY":
            bucket.append(sig)
        else:
            if bucket:
                pairs.append((bucket.pop(0), sig))
    return pairs


def _suggest(stats: dict, strategy: dict | None) -> dict | None:
    if not stats.get("ready") or stats.get("winRate") is None:
        return None
    win_rate = stats["winRate"]
    target = (strategy or {}).get("target") or {}
    stop = (strategy or {}).get("stop") or {}
    ttype = target.get("type")
    tparams = target.get("params") or {}
    stype = stop.get("type")
    sparams = stop.get("params") or {}

    if win_rate < 0.45:
        if ttype == "rr":
            rr = float(tparams.get("rr", 2.0))
            return {
                "type": "tune",
                "reason": f"Win rate is {win_rate:.0%} (<45%) — many trades are being stopped out.",
                "patch": {"target": {"params": {"rr": round(rr + 0.5, 1)}}},
                "explain": f"Raise the risk:reward target to {round(rr + 0.5, 1)}R. "
                           "Fewer trades will hit target, but the wins pay for more losses.",
            }
        if stype == "atr":
            mult = float(sparams.get("mult", 2.0))
            return {
                "type": "tune",
                "reason": f"Win rate is {win_rate:.0%} — the stop may be too tight.",
                "patch": {"stop": {"params": {"mult": round(mult + 0.5, 1)}}},
                "explain": f"Widen the ATR stop multiplier to {round(mult + 0.5, 1)}× "
                           "to give trades more room to breathe.",
            }
    if win_rate > 0.65 and ttype == "rr":
        rr = float(tparams.get("rr", 2.0))
        return {
            "type": "tune",
            "reason": f"Win rate is {win_rate:.0%} (>65%) — winners may be leaving money on the table.",
            "patch": {"target": {"params": {"rr": round(rr + 0.5, 1)}}},
            "explain": f"Stretch the target to {round(rr + 0.5, 1)}R and let winners run further.",
        }
    return None
