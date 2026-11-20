# EXPERIMENTS — what was tried, and what the numbers said

Every entry here is a falsifiable question that was asked of the real code and
real data, with the result kept whether or not it was the answer we wanted. The
point of the file is that **a negative result is a deliverable**: it closes a
branch so nobody spends another week on it.

Rules for adding an entry:

1. State the prediction *before* the run — what would count as success.
2. Report the scoring baseline next to the result. Accuracy without a base rate
   and Brier without climatology are decoration.
3. Record the negative results. They are cheaper to keep than to rediscover.

---

## 1. Regime-conditional forecasting — **NEGATIVE** (closed)

**Question.** A trend rule that works in a calm market and fails in a volatile
one gets blended into a middling coefficient by a single pooled model. Does
fitting one model per volatility regime beat the pooled model?

**Method.** `wand.py forecast --by-regime` on 4,000 hourly SOL-USD bars from
Binance (5.5 months). Regime labels are causal (ATR percentile against its own
trailing 100 bars), the purge is intact, and each regime is scored against *its
own* climatology. A regime with fewer than 60 training rows per fold falls back
to the pooled model and says so.

| variant | Brier | accuracy |
|---|---|---|
| per-regime models | 0.2593 | 51.8% |
| pooled model | **0.2528** | 52.3% |
| climatology (predict the base rate) | 0.2500 | 50.5% |
| pooled + 2 regime dummy columns | 0.2553 | — |

**Result.** Both forms of the idea lose. Splitting by regime divides the training
data three ways and buys variance without buying signal. The motivating lead —
the calm-regime slice at 55.4% accuracy — did not survive a model trained for it
(53.6%); it was 1,214 correlated forecasts in one calm stretch.

**Conclusion.** Do not retry without new features first. `docs/PLAN.md` §2c is
annotated accordingly.

---

## 2. Trend forecasting with 20x more data — **NEGATIVE** (still open)

**Question.** The first evaluation used ~300 bars (12 days). Is the model
skill-less because the sample is tiny?

**Method.** 4,000 bars, walk-forward, purge intact, 3,733 out-of-sample
forecasts, scored against climatology.

| metric | model | climatology | verdict |
|---|---|---|---|
| accuracy | 52.4% | 50.5% | better direction |
| Brier | 0.2526 | **0.2500** | worse probabilities |
| calibration 0.2–0.4 | predicted 0.35 | actual 0.50 | overconfident |
| calibration 0.6–0.8 | predicted 0.65 | actual 0.55 | overconfident |

**Result.** `beats_base` requires both better probabilities and better-than-
coin-flip direction. It has one of the two.

**Conclusion.** More data made the answer clearer, not better. The next untested
ideas are better features (funding, open-interest change, taker imbalance,
cross-asset lead-lag) and better labels (triple-barrier), not more capacity.

---

## 3. "Why not just let deepseek be the forecaster?" — **NEGATIVE** (closed)

**Question.** We have a large model and a 10-parameter logistic regression. Why
train the small one at all?

**Method.** `tools/llm_forecast.py`. Take 250 historical hourly bars from SOL-USD,
present each one to `deepseek-flash` as *numbers only* — trailing returns, ATR%,
RSI, z-scores, range position, volume z — and ask for `{"p_up": ...}`. The asset,
the date and the price level are withheld so memory cannot substitute for
forecasting. Labels are non-overlapping (stride == horizon == 6 bars). Scored
with the same Brier/log-loss code as the trained model.

| | LLM | climatology |
|---|---|---|
| accuracy | 47.6% | **53.2%** |
| Brier | 0.2557 | **0.2490** |
| log loss | 0.7048 | — |
| spread of p_up | mean 0.499, sd **0.048** | — |

**Result.** The model answered ~0.50 every time (sd 0.048) and was *less*
accurate than always predicting the majority direction. It has a prior, not a
view. The 10-parameter logistic model scored 0.2526 on a larger sample — also
worse than climatology, but **better than the LLM**.

Honest caveat: 250 samples gives ±~3pp on accuracy, so "worse than climatology"
is suggestive rather than overwhelming. What is *not* in doubt is the claim that
mattered: the LLM did not beat the baseline, so there is no reason to pay it
$0.03 per 250 forecasts and hours of wall-clock time to do the numeric job.

**Conclusion.** Close it. The LLM reads the world, forms hypotheses, argues and
vetoes — jobs it is good at and where its output can be graded per claim. It does
not produce calibrated probabilities, and no prompt fixes a missing edge.

---

## 4. Better inputs: cross-asset, session, funding — and triple-barrier labels — **NEGATIVE**

**Question.** The model had nine price-only features squeezed out of one symbol's
candles. BTC leads alts, funding says how crowded the leverage is, and
hour-of-day says who is awake. Does adding that — and asking a better-posed
question (did a *tradable* move happen before the stop?) — buy skill?

**Method.** `wand.py forecast --sweep`. Feature groups added cumulatively, both
label types, 4,000 hourly SOL-USD bars, walk-forward with purge intact, each cell
scored against its own label's climatology.

| label | features | n | acc | base | Brier | climo | verdict |
|---|---|---|---|---|---|---|---|
| direction | price only | 3633 | 51.0% | 50.6% | 0.2531 | 0.2500 | no skill |
| direction | +cross | 3633 | 50.5% | 50.6% | 0.2578 | 0.2500 | no skill |
| direction | +session | 3633 | 49.9% | 50.6% | 0.2592 | 0.2500 | no skill |
| direction | +funding | 3600 | 49.6% | 50.7% | 0.2641 | 0.2500 | no skill |
| **triple** | **price only** | 3633 | **52.0%** | 50.9% | **0.2511** | 0.2499 | no skill |
| triple | +cross | 3633 | 51.7% | 50.9% | 0.2562 | 0.2499 | no skill |
| triple | +session | 3633 | 51.3% | 50.9% | 0.2571 | 0.2499 | no skill |
| triple | +funding | 3600 | 50.9% | 51.0% | 0.2588 | 0.2499 | no skill |

**Result.** Two separate findings.

1. **The triple-barrier label is a better-posed question than direction** — best
   Brier of any cell (0.2511 vs 0.2531) and the best accuracy (52.0%). It is
   still worse than climatology, so it does not pass, but the label is not the
   problem; it is the closest thing to a signal found so far.
2. **Every added feature group made it worse, monotonically, on both labels.**
   Direction: 0.2531 → 0.2578 → 0.2592 → 0.2641. Triple: 0.2511 → 0.2562 →
   0.2571 → 0.2588. Six consecutive degradations is the signature of pure
   overfitting: real columns, no incremental signal, more variance to fit.

**Conclusion (revised after entry 5).** On 166 days the added features lost
badly. On 2.3 years they lose only slightly (0.2520 vs 0.2511 price-only), so the
size of that degradation was itself a small-sample effect. The verdict does not
change — no skill either way — but the *reason* does: it is not that the features
are wrong, it is that nothing here beats climatology.

**A bug found while re-running this on more data.** `load_funding` asked the
venue for its default page and got 500 records — 137 days — then paged forward
from the end, which returns nothing. The "two years of features" run was silently
evaluating 137 days. Fixed to walk a window with `startTime`/`endTime` and now
pulls 2,400 records (2.2 years). A feature that quietly covers a fifth of the
data looks exactly like a feature that doesn't work.

---

## 5. Twenty times more history — **NEGATIVE** (branch closed)

**Question.** Every model so far used 166 days. Is the missing skill simply a
missing sample?

**Method.** 19,999 hourly SOL-USD bars from Binance, 2024-08-08 → 2026-11-19
(2.3 years), walk-forward with purge intact, refit every 500 bars, scored against
climatology. Two labels, with and without the full feature set.

| label | features | n | acc | base | Brier | climo | verdict |
|---|---|---|---|---|---|---|---|
| direction | price only | 19,333 | 50.4% | 50.2% | 0.2511 | 0.2500 | no skill |
| direction | cross+session+funding | 18,500 | **51.2%** | 50.2% | 0.2520 | 0.2500 | no skill |
| triple | price only | 19,333 | 49.9% | 50.9% | **0.2510** | 0.2499 | no skill |
| triple | cross+session+funding | 18,500 | 49.9% | 51.0% | 0.2524 | 0.2499 | no skill |

**Result.** 20x the data moved the best Brier from 0.2511 to 0.2510. It did not
move it past 0.2500. Direction accuracy with the full feature set reaches 51.2%
against a 50.2% base rate — a +1.0pp edge in direction with *worse* probabilities,
which is exactly the pattern `beats_base` exists to reject: you can be
slightly-more-often right and still be worse calibrated than a constant.

**Conclusion.** The sample was not the bottleneck. 2.3 years of hourly bars is
enough to learn from if there were something to learn. What is left is not more
data and not more columns: it is a different question — a different timeframe
(4h/1d), a longer horizon (24 bars), or a different market. Testing another
feature on another two years would be the same experiment with a new noun.

---

## 6. Different timeframe and horizon — **NEGATIVE** (branch closed)

**Question.** The last untested lever: the 6-bar horizon on 1h bars may be the
wrong question. A day, or a week, or a coarser bar, might have signal that an
hourly close does not.

**Method.** `wand.py forecast --timeframe … --horizon …`, price features only,
walk-forward with purge intact:

| timeframe | horizon | n | acc | base | Brier | climatology |
|---|---|---|---|---|---|---|
| 1h | 24 bars (1 day) | 9,315 | 49.7% | 52.5% | 0.2557 | 0.2494 |
| 4h | 6 bars (1 day) | 5,333 | 47.9% | 50.6% | 0.2517 | 0.2500 |
| 1d | 6 bars (6 days) | 833 | 52.2% | 54.0% | 0.2597 | 0.2484 |

**Result.** All three below climatology on Brier, and all three *below their own
base-rate accuracy*. Coarser bars and longer horizons do not reveal the signal;
if anything the 4h series (which gets more information per bar for the same wall
time) is the worst of the three.

**A crash found on the way.** `--timeframe 4h` died with a KeyError:
`binance_candles` used Coinbase's granularity map to compute bar length, and that
map has no 4h because Coinbase does not serve it. Split into `_BAR_SECONDS` (bar
length, all venues) and `_GRANULARITY` (Coinbase's argument whitelist). The
silent version of this bug would have been worse: an unknown timeframe quietly
defaulting to one hour would have mislabelled every bar.

**Conclusion.** Six branches closed on the same question. 1h/4h/1d price action
on SOL does not predict its own next day, and the honest conclusion is that this
edge is not in this data at this resolution. What remains is a different KIND of
question — structural rather than predictive (funding carry / basis, liquidation
cascades) — or a different market.

---

## 7. Funding-rate carry — **NEGATIVE** (but the first question of a different kind)

**Question.** Every branch so far asked "which way will price go". This one asks
whether you can be paid for taking the other side of a persistent imbalance:
hold spot long + perp short, collect the funding rate, have no view on price.

**Method.** `wand.py carry` on 800 days of 8h funding (2,400 periods,
2024-09 → 2026-11). Funding is collected on the perp leg at the published rate;
both legs are marked separately so the basis is its own P&L line; fees are
charged on all four legs of a round trip (0.10% spot + 0.05% perp taker, plus
2 bps slippage per leg = 38 bps); decisions use only funding already published.

| run | held | funding | basis | costs | net | annualised |
|---|---|---|---|---|---|---|
| enter ≥1bp avg, exit <0 | 25% | +499 bps | −1376 bps | −988 bps | **−1865 bps (−18.7%)** | −8.5% |
| hold the whole window | 100% | +325 bps | +41 bps | −38 bps | **+328 bps (+3.3%)** | **+1.5%** |

**What the data says.** SOL funding averaged **+0.135 bps per 8h** over the
window — 62% of periods positive, **325 bps total across 800 days** if you never
churn. One round trip costs 38 bps. Meanwhile the mark-vs-spot basis has a
**standard deviation of 84 bps** and a range of −862 to +697 bps: every entry and
exit is a coin flip on a move twice the size of the entire round-trip cost, and
26 round trips turned +499 bps of funding into −1865 bps of P&L.

**Conclusion.** The best possible case — perfect hold, no churn, no liquidation,
no custody risk, one venue — is **+1.5% a year**, which is below the risk-free
rate while carrying exchange and margin risk. Any realistic entry/exit rule
destroys it. Carry is real in markets where funding is large and persistent; on
SOL at this size it is not.

**Two bugs found by hand-checking, both flattering the strategy:**

1. The short perp leg was booked as `M0/M1 - 1` instead of `1 - M1/M0`. The
   reciprocal measures the move against the *exit* notional, and by AM-GM the
   two legs' ratios sum positive — so a delta-neutral position "earned" +2095 bps
   of phantom basis while SOL fell 48%. That is the difference between a −19%
   result and a +6% one.
2. A position still open at the end of the window was never closed, hiding its
   exit fee and final basis.

Nine tests in `test_carry.py` now pin the arithmetic by hand: a neutral position
must show zero basis P&L whether price crashes or rallies, a premium entry must
be credited its convergence, the window-end close must be charged, and a funding
spike cannot pull in an entry before it existed.
