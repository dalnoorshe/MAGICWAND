# MEMORY — how to work on this project

Durable notes for whoever (or whatever) picks this up next, including future
sessions of the agent that built it. Written down because context is lost and
these lessons were expensive.

---

## The one command to run before you trust anything

```bash
python3 wand.py audit                     # both angles, ~4s
python3 wand.py audit --angle durability  # state / restarts / accounting only
python3 -m unittest discover -s tests -v  # 464 tests
```

When a panel says a source is unavailable, run `python3 wand.py sources` before
touching any parsing code: it separates "our code is broken" from "the internet
said no". Reddit blocks whole networks, so a FAIL there is usually the network.

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
  every refresh; a README claiming 179 tests). Verify with a test or a grep
  after editing, not by assuming the edit applied. The same applies to an edit
  that was *asserted but never written*: two doc edits in this project built the
  replacement, asserted the anchor matched, then wrote a different variable — the
  prose silently vanished while the script printed success. Grep for the new text
  afterwards; a passing assert is not a write.
- **Never show a replay without its age, and never hide work in progress.** The
  research panel renders the last cycle out of the ledger, so it is full the
  instant the page loads — which looks exactly like a fabricated or
  instantaneous result. Every replayed reading now carries "researched 5 min
  ago" and a stale warning past 2.5 tick intervals; while a cycle runs,
  `data/research-inflight.json` drives a live roster that fills in reader by
  reader ("collecting market data", then each reader, then "weighing the
  verdicts"). The marker is written at stage boundaries, cleared in a `finally`,
  and ignored if absent, corrupt, in the future, or older than 15 minutes. A
  panel that cannot say when it last looked, or what it is doing now, is not
  trustworthy however correct its numbers are.
- **A UI bug is not cosmetic.** One undefined variable aborted the whole refresh
  loop, so autopilot, audit, storage and the feed all looked dead while the
  header still updated. Every panel now renders inside its own guard, and
  `tests/test_ui_runtime.py` executes the real JS against a stub DOM.
- **Never score a reader on a raw hit rate.** Over the graded sample 76% of 24h
  windows fell and every reader leaned bearish, so "always bearish" scored 76%
  for free. Two readers sat *below* that baseline while carrying influence
  0.62/0.60, and `charts` scored 88% on 26 claims and got 0.96. Real samples are
  also clustered: 575 grades sat on 31 distinct price paths. `skill = hit rate −
  base rate for the direction called`, counted in distinct paths, is what drives
  influence (`research_loop.agent_track_record`, used by both the synthesizer
  and `reflect` — do not fork this arithmetic again; it was forked once and the
  two copies drifted).
- **Coverage is part of the evidence.** Every `Source` declares its providers and
  every verdict carries the fraction that answered. `synthesize` shrinks the
  evidence by it, charges the denominator for readers that never reported, and
  raises the veto bar for thin channels. Before this, a dead source dropped out
  of the average and the survivors renormalised to 100% — the decision changed
  without getting any weaker. The operator caught that by eye;
  `tests/test_coverage.py` now catches it by machine.

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
| `magicwand/compose.py` | strategy grammar: composed ideas, same gate |
| `docs/PLAN.md` | deferred work, each with its trigger |
| `docs/EXPERIMENTS.md` | falsified ideas with their numbers — read before retrying |
| `tools/llm_forecast.py` | scores the LLM against climatology; 3 cents per run |
| `docs/PLATFORMS.md` | broker integrations, MT5 caveat |

## State of play (as of the last session)

- 464 tests, 0 audit gaps, 23 audit passes.
- The validation gate has approved **0** strategies: at ~1.2% round-trip costs
  nothing has survived yet. That is the honest blocker, not a bug.
- At $50 equity the desk can place **no** trade: $0.50 per trade against a $5
  venue minimum. ~$500 of equity is needed before paper trading means anything.
- No live broker exists by design. `live` tier is refused.
- Proposals were 61% identical reruns until exploration was tightened; the
  funnel card in the console tracks it.
- **Reddit refuses scripted clients** (403/404/429, hour by hour), and its feed
  path was wrong (`/new/.rss` 404s; `/new.rss` is the feed). Social is now
  crowd attention (CoinGecko trending) + crowd mood (Fear & Greed) + Reddit
  best-effort behind a 45-minute cooldown. The prompt says which part was
  missing, so a reader cannot describe posts it never saw. Do not make any
  single external source load-bearing again — that is how one 404 greyed out a
  whole reader.
- **No reader has demonstrated skill.** Against a per-horizon base rate on the
  real ledger: `charts` +14.8pp on 12 independent windows (not significant),
  `onchain` +4.8pp, `social` +9.5pp, `derivatives` −13.5pp, `news` −13.2pp.
  Every influence sits in 0.38–0.50. The loop now measures honestly; it has not
  yet found anything worth trusting. That is the real blocker behind the gate's
  zero passes, and no amount of extra model calls substitutes for it.
- The trend forecaster reports **no skill** on real SOL data (50.1% vs 50.9%
  base rate, Brier worse). Do not "improve" it by weakening the baseline: the
  first version compared against a hard 0/1 prediction and claimed an edge it
  did not have. Baselines are climatology + coin flip, and `beats_base` needs
  both. Next steps are better FEATURES and LABELS, not a bigger model
  (docs/PLAN.md 2c). Re-run on 4,000 bars (5.5 months; 3,733 out-of-sample
  forecasts) it is still **no skill**: accuracy 52.4% vs 50.5% climatology, but
  Brier 0.2526 vs 0.2500 — better direction, worse probabilities, uncalibrated
  at both extremes (predicted 0.2–0.4 → actual 0.50; predicted 0.6–0.8 → actual
  0.55). The calm-regime slice (n=1,214, 55.4%) was the only lead, and it died
  under a dedicated model (53.6%).
- **Regime training is a dead end, tested twice.** One fit per volatility regime
  (`wand.py forecast --by-regime`): Brier **0.2593 vs 0.2528 pooled**, and two of
  the three regimes had too little data and fell back anyway. The cheap form —
  pooled model plus two regime dummy columns — also lost: **0.2553 vs 0.2528**.
  Splitting by regime divides the data and buys variance without buying signal.
  Do not retry without new features first; docs/PLAN.md §2c is annotated.
- **"Why not just use the LLM to forecast?" is answered, negatively.** 250
  historical setups given to `deepseek-flash` as numbers only (no asset, no date,
  so it cannot recall the outcome): it answered ~0.50 every time (sd 0.048),
  accuracy 47.6% against a 53.2% base rate, Brier 0.2557 vs 0.2490 climatology.
  3 cents and 6 minutes. The 10-parameter regression beat it. The LLM's job is
  research and judgement, where each claim can be graded — not calibrated
  probabilities. Reproduce with `python3 tools/llm_forecast.py`.
- `docs/EXPERIMENTS.md` is where falsified ideas go, with their numbers, so
  nobody re-runs them. Two branches are closed there (regime training, LLM
  forecasting); features and labels remain open.
- Backtests are linear again. `decide` gets a growing prefix each bar to make
  lookahead impossible, and every indicator series was being rebuilt per prefix
  (O(n²)): 300 bars took 1s, 4,000 took 45s. The series are trailing-window
  functions, so one pass over the full list is already causal; 4,000 bars now
  take 0.08s with identical trades. The whole suite went from 73s to 48s.

## Style that has worked here

- Small honest numbers beat impressive ones. Report the failure loudly.
- When something is deferred, write the trigger that would make it correct.
- Prefer a test that would have caught the bug over a paragraph explaining it.
- Answer the question that was asked; a status dump is not an answer.
