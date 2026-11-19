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
