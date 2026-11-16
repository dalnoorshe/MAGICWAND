# MEMORY — how to work on this project

Durable notes for whoever (or whatever) picks this up next, including future
sessions of the agent that built it. Written down because context is lost and
these lessons were expensive.

---

## The one command to run before you trust anything

```bash
python3 wand.py audit                     # both angles, ~4s
python3 wand.py audit --angle durability  # state / restarts / accounting only
python3 -m unittest discover -s tests -v  # 383 tests
```

`wand audit` runs adversarial scenarios against the real code in throwaway
directories. **It is part of the routine**: run it after touching the tick loop,
the desk, the risk engine, the ledger, or anything that parses prices.

It saves its result to `data/audit.json`; the console shows the last run.

---

## Non-negotiable invariants

These were each learned by breaking them. Do not "improve" them away.

1. **The model judges; the operator decides.** An LLM may propose a strategy, a
   market or an answer. It may not place a trade, change a tier, change a market
   or edit settings. The chat is read-only *by construction* (it imports none of
   desk/broker/ledger) — keep it that way.
2. **Veto only.** A research reader can block a trade and can never open one.
3. **Risk ceilings come from the tier, not the AI.** A human sets the tier; the
   gatekeeper enforces it mechanically on every order.
4. **Fail closed.** Corrupt or missing state must reduce authority, never grant
   it. A corrupt `policy.json` = research tier. A corrupt `desk.json` = no
   trading, file preserved.
5. **Never guess a balance.** If a broker is connected and cannot report equity,
   raise — do not substitute the configured value.
6. **Costs are real.** Fills are pessimistic, fees and slippage apply on both
   sides, stops win ties, and the buy-and-hold comparison is exposure-adjusted.
7. **No lookahead.** A decision on bar *i* fills at the open of bar *i+1*. The
   forming candle is never drawn or traded.
8. **Everything is recorded.** Ledger appends are hash-chained and file-locked;
   every refusal carries its reason.
9. **Bars are not ticks.** `max_hold_bars` counts bars; positions age when the
   bar timestamp advances.
10. **Non-finite numbers are poison.** NaN comparisons are all `False`, so NaN
    slips past every `<` guard. Reject at Bar, RiskEngine and Broker.

## Do not do this

- **Never run `rm` against `data/`** while a server may be running, and never let
  a test or probe write to the real `data/`. Use a temp root. (This was done
  once; it deleted the operator's chat history.)
- **Never hardcode a model list, a market list, or a fee assumption** where a
  fetched or configured value exists.
- **Never add a vendor, database or cloud service** without being asked. The
  whole design is local files.
- **Never let a test depend on network access** (Reddit rate-limits; exchanges
  vary). Mark live tests and skip by default.
- **Never weaken the validation gate** to make a strategy pass.
- **Never trust a silent `str.replace`.** Twice an edit did nothing because the
  indentation or spacing differed, and the result shipped (a UI that threw on
  every refresh; a README claiming 383 tests). Verify with a test or a grep
  after editing, not by assuming the edit applied.
- **A UI bug is not cosmetic.** One undefined variable aborted the whole refresh
  loop, so autopilot, audit, storage and the feed all looked dead while the
  header still updated. Every panel now renders inside its own guard, and
  `tests/test_ui_runtime.py` executes the real JS against a stub DOM.

## Where things live

| Path | What |
|---|---|
| `magicwand/runner.py` | the tick: data → research → reflect → propose → maybe trade |
| `magicwand/desk.py` | sandboxed sub-account; the only place a position opens |
| `magicwand/gatekeeper.py` | risk tiers and the mechanical order check |
| `magicwand/risk.py` | sizing, limits, kill switch |
| `magicwand/ledger.py` | hash-chained record |
| `magicwand/backtest.py` | honest backtester + reality check |
| `magicwand/propose.py` | strategy proposals, novelty, validation gate |
| `magicwand/reflect.py` | bounded self-tuning, family demotion |
| `magicwand/audit.py` | the scenarios |
| `magicwand/account.py` | broker → paper → config balance resolution |
| `magicwand/chat.py` + `web.py` | read-only operator chat with search/fetch |
| `magicwand/forecast.py` | walk-forward trend model, purged, honestly scored |
| `docs/PLAN.md` | deferred work, each with its trigger |
| `docs/PLATFORMS.md` | broker integrations, MT5 caveat |

## State of play (as of the last session)

- 383 tests, 0 audit gaps, 23 audit passes.
- The validation gate has approved **0** strategies: at ~1.2% round-trip costs
  nothing has survived yet. That is the honest blocker, not a bug.
- At $50 equity the desk can place **no** trade: $0.50 per trade against a $5
  venue minimum. ~$500 of equity is needed before paper trading means anything.
- No live broker exists by design. `live` tier is refused.
- Proposals were 61% identical reruns until exploration was tightened; the
  funnel card in the console tracks it.
- The trend forecaster reports **no skill** on real SOL data (50.1% vs 50.9%
  base rate, Brier worse). Do not "improve" it by weakening the baseline: the
  first version compared against a hard 0/1 prediction and claimed an edge it
  did not have. Baselines are climatology + coin flip, and `beats_base` needs
  both. Next steps are better FEATURES and LABELS, not a bigger model
  (docs/PLAN.md 2c).

## Style that has worked here

- Small honest numbers beat impressive ones. Report the failure loudly.
- When something is deferred, write the trigger that would make it correct.
- Prefer a test that would have caught the bug over a paragraph explaining it.
- Answer the question that was asked; a status dump is not an answer.
