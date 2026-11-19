"""EXPERIMENT — can the LLM beat the base rate at forecasting?

The standing question: if we have deepseek, why train a tiny model at all?
The honest way to settle it is to score the LLM the same way everything else in
this project is scored: Brier against climatology, on labels it never saw.

    python3 tools/llm_forecast.py --samples 250

Design notes that matter:

  * No symbol, no dates, no prices. Only relative numbers (percent changes,
    ATR as a % of price, RSI, z-scores, volume z) so the model cannot recall
    "what SOL did in November 2026" from memory instead of forecasting.
  * Labels are non-overlapping: the sample stride equals the horizon, so one
    observation cannot leak into the next.
  * The question is a probability, in JSON, so it can be scored rather than
    narrated.
  * It is scored against the SAME climatology baseline the trained model faces.

Costs real money (a fraction of a cent per call). It is not part of the bot and
never trades; it exists to answer one question with a number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import data as data_mod                  # noqa: E402
from magicwand import forecast as fc                    # noqa: E402
from magicwand.config import Config                     # noqa: E402
from magicwand.llm import DeepSeekClient, LLMError      # noqa: E402
from magicwand.settings_store import SettingsStore, resolve_key  # noqa: E402

SYSTEM = (
    "You are a quantitative forecaster. You will be given a compact numerical "
    "summary of an unknown liquid crypto asset's recent hourly bars: trailing "
    "returns, volatility, RSI, z-scores against moving averages, range position "
    "and volume z-score. You are NOT told the asset, the date, or the price "
    "level, so memory cannot help; only the numbers.\n"
    "Estimate the probability that the close 6 hourly bars from now is HIGHER "
    "than the current close.\n"
    'Reply with JSON only: {"p_up": <number between 0 and 1>}'
)


def _context(bars, i: int, horizon: int) -> tuple[str, int]:
    f = fc.features_at(bars, i)
    if f is None:
        raise ValueError("no features")
    rets = [round((bars[k].close / bars[k - 1].close - 1) * 100, 3)
            for k in range(i - 23, i + 1)]
    lines = [
        "last 24 hourly returns (%), oldest first: " + ", ".join(map(str, rets)),
        "features: " + json.dumps({k: round(f[k], 4) for k in fc.FEATURES}),
        f"horizon: {horizon} hourly bars",
    ]
    y = 1 if bars[i + horizon].close > bars[i].close else 0
    return "\n".join(lines), y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOL-USD")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--candles", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--samples", type=int, default=250)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    key = resolve_key(Path("."))
    if not key:
        print("no API key — set DEEPSEEK_API_KEY or save one in the console")
        return 2
    saved = SettingsStore(Path("data/settings.json")).load()
    model = args.model or saved.model or "deepseek-flash"
    client = DeepSeekClient(api_key=key, model=model, thinking="disabled",
                            max_calls_per_day=max(args.samples + 50, 400),
                            max_cost_per_day_usd=3.0)

    bars = data_mod.fetch(args.exchange, args.symbol, "1h", args.candles)
    horizon = args.horizon
    lo = 200
    hi = len(bars) - horizon - 1
    # Stride == horizon keeps labels from overlapping.
    idxs = list(range(hi, lo, -horizon))[: args.samples]
    idxs.reverse()
    print(f"{len(bars)} bars · {len(idxs)} samples · stride {horizon} bars "
          f"· model {model}")

    ps: list[float] = []
    ys: list[int] = []
    failed = 0
    for n, i in enumerate(idxs, 1):
        try:
            user, y = _context(bars, i, horizon)
        except (ValueError, IndexError):
            continue
        try:
            raw = client.chat(system=SYSTEM, user=user, temperature=0.0,
                              max_tokens=60, json_mode=True)
            from magicwand.jsonfix import parse_json_lenient
            parsed = parse_json_lenient(raw)
            if not parsed.ok or not isinstance(parsed.obj, dict):
                failed += 1
                continue
            p = float(parsed.obj.get("p_up"))
        except (LLMError, ValueError, TypeError, KeyError, IndexError):
            failed += 1
            continue
        if not (0.0 <= p <= 1.0):
            failed += 1
            continue
        ps.append(p)
        ys.append(y)
        if n % 25 == 0:
            print(f"  {n}/{len(idxs)} · running Brier "
                  f"{fc._brier(ps, ys):.4f} · cost ${client.usage.cost_usd:.3f}")

    if len(ps) < 30:
        print(f"only {len(ps)} usable forecasts ({failed} failed) — aborting")
        return 1

    base = sum(ys) / len(ys)
    climo = base * (1.0 - base)
    acc = sum(1 for p, y in zip(ps, ys) if (p >= 0.5) == (y == 1)) / len(ys)
    llm_brier = fc._brier(ps, ys)
    print()
    print("  LLM FORECAST — scored like everything else")
    print("  " + "-" * 66)
    print(f"  samples          {len(ps)} ({failed} failed)")
    print(f"  base rate        {base:.1%}")
    print(f"  accuracy         {acc:.1%}  (climatology "
          f"{max(base, 1 - base):.1%})")
    print(f"  Brier            {llm_brier:.4f}")
    print(f"  climatology      {climo:.4f}")
    print(f"  log loss         {fc._logloss(ps, ys):.4f}")
    print(f"  mean p_up        {statistics.mean(ps):.3f} "
          f"(sd {statistics.pstdev(ps):.3f})")
    verdict = ("BEATS climatology" if llm_brier < climo
               else "NO skill: worse probabilities than predicting the base rate")
    print(f"  verdict          {verdict}")
    print(f"  cost             ${client.usage.cost_usd:.4f} "
          f"over {client.usage.calls} calls")
    print()
    return 0 if llm_brier < climo else 1


if __name__ == "__main__":
    raise SystemExit(main())
