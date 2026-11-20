# MEMORY — how to work on this project

Durable notes for whoever (or whatever) picks this up next, including future
sessions of the agent that built it. Written down because context is lost and
these lessons were expensive.

---

## The one command to run before you trust anything

```bash
python3 wand.py audit                     # both angles, ~4s
python3 wand.py audit --angle durability  # state / restarts / accounting only
python3 -m unittest discover -s tests -v  # 525 tests
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

11. **An unproven edge sizes to zero.** `magicwand/sizing.py` sizes from the
    LOWER confidence bound of the measured per-trade edge, net of costs, at
    half-Kelly with a hard cap. When the bound does not cover costs the size is
    exactly 0.0 — not small. `RiskEngine.size_entry(edge=...)` refuses the order
    and says the arithmetic. Without an edge argument the old fixed ceilings
    still apply, which is why they were conservative: they stood in for
    evidence.

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
| `magicwand/features.py` | cross-asset / session / funding features + triple-barrier labels |
| `magicwand/carry.py` | funding-carry simulation: two legs, basis, four-leg fees |
| `magicwand/sizing.py` | Kelly on the conservative edge; zero when unproven |
| `magicwand/journal.py` | paper-run scorecard; outages are not silence |
| `magicwand/compose.py` | strategy grammar: composed ideas, same gate |
| `docs/PLAN.md` | deferred work, each with its trigger |
| `docs/EXPERIMENTS.md` | falsified ideas with their numbers — read before retrying |
| `tools/llm_forecast.py` | scores the LLM against climatology; 3 cents per run |
| `docs/PLATFORMS.md` | broker integrations, MT5 caveat |

## State of play (as of the last session)

- 525 tests, 0 audit gaps, 23 audit passes.
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
- **2.3 years did not help either.** 19,999 hourly bars (2024-08 → 2026-11),
  refit every 500 bars: best Brier 0.2510 (triple, price only) vs climatology
  0.2499. Direction accuracy with all features 51.2% vs 50.2% base — a +1pp edge
  with worse probabilities, which is exactly what `beats_base` rejects. 20x more
  data moved the number by 0.0001. The sample was not the bottleneck.
- **Better inputs did not buy skill either — they made it worse.** Cross-asset
  (BTC/ETH returns, BTC-SOL correlation, relative strength), session (hour-of-day)
  and funding (rate, z-score, change) added cumulatively on 4,000 bars: Brier
  0.2531 → 0.2578 → 0.2592 → 0.2641 for direction, and 0.2511 → 0.2562 → 0.2571
  → 0.2588 for triple-barrier. Six monotone degradations = overfitting, not a
  bug. The triple label is the best-posed question found so far (0.2511, 52.0%)
  but still loses to climatology. The bottleneck is the SAMPLE: 166 days ≈ 600
  semi-independent outcomes, and 19 columns cannot learn from that. Reproduce
  with `wand.py forecast --sweep`. New feature modules live in
  `magicwand/features.py` (causal, timestamp-aligned, holes are dropped not
  imputed). On 2.3 years instead of 166 days they lose only slightly
  (0.2520 vs 0.2511), so the earlier size of the degradation was itself a
  small-sample effect. Verdict unchanged: no skill.
- **`load_funding` was silently covering 137 days, not two years.** The venue's
  no-start page returns 500 records; paging forward from the last one returns
  nothing. A feature that quietly covers a fifth of the data looks exactly like a
  feature that does not work — and it produced a wrong conclusion that had to be
  retracted. It now walks a `startTime`/`endTime` window (2,400 records, 2.2y).
- **Timeframe and horizon do not rescue it either.** 1h/h24 (0.2557 vs 0.2494),
  4h/h6 (0.2517 vs 0.2500), 1d/h6 (0.2597 vs 0.2484): all below climatology on
  Brier AND below their own base-rate accuracy. Also fixed a KeyError:
  `binance_candles` used Coinbase's granularity map for bar length and that map
  has no 4h. Now `_BAR_SECONDS` (all venues) vs `_GRANULARITY` (Coinbase
  whitelist only).
- **Funding carry is real but too small here.** 800 days, 2,400 funding periods:
  SOL funding averages +0.135 bps per 8h, +325 bps total if you never churn. One
  round trip costs 38 bps, and the mark-vs-spot basis has sd 84 bps. Perfect hold
  = +1.5% a year (below risk-free, before exchange/margin/liquidation risk); any
  churn rule = −19%. `wand.py carry`. Two bugs found by hand-checking both
  flattered it: the short leg was booked as `M0/M1-1` instead of `1-M1/M0` (which
  fabricates P&L from volatility), and the window-end close was never charged.
- **The paper journal exists, and it catches outages.** `wand.py journal` reports
  per-day ticks/research/trades/refusals and — the point — a day with no ledger
  entries at all is labelled "NOTHING RAN — outage, not a refusal". Without
  that, a run that was off for a week reads as six calm days of discipline. The
  current real data shows 6 active days, 0 trades, 0 refusals, 234 research
  cycles, $0.70 spent, and **3 outage days** (11-14, 11-15, 11-17).
- **`--equity` on serve/loop** sets an explicit paper balance and writes a `note`
  to the ledger saying it is an assumption, not a measurement. At $500 the
  risk engine can clear the $5 venue minimum; at the default $50 it cannot.
- **Sizing is now evidence-gated.** `wand.py edge` measures the strategy's
  per-trade edge and prints the size it justifies. On the default strategy:
  −114.5 bps mean, −165.4 bps conservative, 140 bps round trip → **0.00% of
  equity**. A unit bug was caught doing this: `Trade.return_pct` is FRACTIONAL,
  so bps is ×10,000 not ×100 — the wrong factor understated a −114 bps/trade
  loser as −1.14 bps and would have argued FOR a position. `edge_from_trades`
  now owns that conversion and two tests pin it.
- **The 4×3 scan is the decisive one: 0/12 cells beat climatology.** `wand.py
  scan` (markets × horizons, triple label, chance-corrected): zero cells beat
  their Brier baseline where a coin flip would give ~6, zero at p<0.05 raw
  (0.6 expected), none survive Bonferroni. Every cell is 0.0015–0.0037 worse
  than its own climatology — the exact signature of a no-signal model paying a
  parameter-fitting variance tax. That is now the strongest statement in the
  repo: the price-action features carry nothing across BTC/ETH/PAXG/SOL at
  6/24/72 bars. Test cells are correlated, so the correction is generous.
- `docs/EXPERIMENTS.md` is where falsified ideas go, with their numbers, so
  nobody re-runs them. Closed: regime training, LLM forecasting, better
  features, better labels, more history, timeframe/horizon, funding carry.
  Still open: liquidation-cascade reversion, a different market, and
  calibration as a post-process.
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
