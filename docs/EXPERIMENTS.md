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

**Conclusion.** On 166 days of 1h bars with a 6-bar horizon — roughly 600
semi-independent outcomes — a 19-column model cannot learn anything the market
is not handing out. The bottleneck is now the SAMPLE, not the feature list or
the label. Next: years of history (Binance pages 1,000 bars per request, so
20,000+ hourly bars is a few seconds of fetching), higher timeframes, and longer
horizons. Adding another clever feature to 166 days would be the same mistake
with a new name.
