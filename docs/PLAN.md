# PLAN — decisions deferred, each with its trigger

Not a wishlist. Each item here is something deliberately *not* built, with the
condition that would make building it correct. If the trigger never fires, the
work was never needed — which is the point.

---

## 1. Surface the best candidate's numbers when one exists

**Status:** not built.
**Trigger:** the first strategy actually passes the validation gate.

Today the chat and the console say "the gate has approved 0 strategies", which
is true and uninteresting. Once one passes, that line becomes the most valuable
thing on the screen: which family, what out-of-sample return, how many trades,
what the holdout looked like.

Building it now would mean designing and testing a view for a state that has
never occurred. The plan is to show, per approved strategy: family, hypothesis,
train vs holdout return, trade count, t-stat, maximum drawdown, and the date it
was approved — plus a warning when the approval is older than its own evidence.

**Why not now:** there is nothing to display, and a leaderboard of zero teaches
the wrong thing (it looks like a target to fill).

---

## 2. Why proposals die — done, in the console

**Status:** **built** (this was item 1's smaller sibling and it had real data).

The "Proposal funnel" card shows pitched / reached-the-gate / passed / identical
rerun / near-duplicate / malformed, plus which gates fail most. The first look
was informative: 141 proposals, only **7** ever reached the gate, 87 were
identical reruns. That is a proposer problem, not a strategy problem, and it was
invisible before.

---

## 2b. Tools for the voting agents (search + fetch)

**Status:** half built. The **chat** now has web tools (search + fetch, cited,
budgeted, untrusted-fenced). The four readers and the chart reader do **not** —
they still see only the snapshot.

**Why not yet, and what I think the right shape is.** Giving every reader a tool
loop inside every tick would: cost money every 5 minutes, add tens of seconds of
latency per cycle, make the same snapshot produce different research on different
runs, and multiply the prompt-injection surface by the number of pages fetched.
It would also make "why did it veto?" much harder to answer.

The shape that keeps the good parts:

| Tier | What it has | When it runs |
|---|---|---|
| fast readers | the snapshot only | every tick, as now |
| **deep researcher** | search + fetch, max ~5 calls | scheduled (hourly), or when the fast readers disagree |

The deep researcher would return the same verdict schema **plus citations**, so
its claims grade exactly like everyone else's and the ledger can show what it
read. It stays **veto-only** — tools never grant trading authority.

**Trigger:** the fast readers are producing verdicts consistently (no DEGRADED
readers for a week) and the operator wants deeper research than a fixed snapshot
allows.

---

## 2d. Composition vs secrecy

**Composition is built.** `wand compose` lets the model invent a strategy from
an audited vocabulary — up to three ANDed conditions over nine features, plus
ATR stop/target and a time stop — and the result goes through the *same*
validation gate as any catalog template. An invented idea earns its place with
evidence, not novelty.

**Secrecy is not.** A strategy that trades is observable in its fills; one that
runs on a machine you control is readable by whoever has that machine. Obfuscating
the code buys effort, not safety, and it makes the thing unauditable — which is
the opposite of this project's value.

What actually protects an edge, in order of how much it helps:

1. **The graded record.** Months of dated, hash-chained claims with outcomes is
   not copyable by reading code — it is accumulated evidence tied to your ledger.
2. **Data and feature work.** The edge, if any, lives in inputs and labels, not
   in a clever expression.
3. **Operational hygiene.** Local-only state, trade-only keys, no publishing of
   the strategy spec, disk encryption if the machine leaves your control.
4. **Speed of iteration.** A record that keeps being re-validated is harder to
   copy than a static trick.

**Trigger to revisit:** the operator wants to run this for someone else, which
raises a different question (licensing and disclosure) than keeping it private.

---

## 2c. Trend forecasting: what the first result says

**Status:** built (`wand forecast`), and it reports **no skill** on real data.

The honest numbers on 2,733 out-of-sample SOL/USDT hourly forecasts (4 months,
6-bar horizon, purged walk-forward):

| | model | climatology |
|---|---|---|
| accuracy | 50.1% | 50.9% |
| Brier | 0.2564 | 0.2499 |
| log loss | 0.7076 | — |

Worse than predicting the base rate every time. Calibration was also inverted at
the top: forecasts of 0.65 resolved 0.49. That is not a bug in the code — it is
what a small feature set on a nearly-efficient 1h series looks like.

**What would give it a real chance** (each is a contained experiment, and each
must be scored the same way):

1. **Better features** — funding rate, open-interest change, taker buy/sell
   imbalance, BTC dominance, cross-asset lead-lag (BTC usually leads alts),
   time-of-day/session effects. The bars alone throw away most of the signal.
2. **Better labels** — triple-barrier (did it hit +x% before −y%?) instead of
   fixed-horizon direction, which is noisy and pays no attention to path.
3. **Regime-conditional models** — train per volatility regime rather than one
   model with a regime feature averaged away.
4. **Calibration** — Platt/isotonic on a held-out slice; the model is
   overconfident at its extremes.
5. **Longer history and higher timeframes** — 6h/daily series are less noisy,
   and there is now a Binance feed with thousands of bars.

**What is NOT the answer:** a bigger model. On this data a deep net would fit
noise more efficiently, and the Brier score would say so.

**Trigger to continue:** the operator wants to spend time on features 1 and 2,
which are the two with the best expected return.

---

## 3. Multi-market, simultaneously

**Status:** not built. One market per instance.
**Trigger:** (a) the single-market pipeline shows a positive, graded record, and
(b) a second market is wanted for diversification rather than novelty.

Needs: portfolio state instead of one position, per-symbol risk, per-market
ledger attribution, a multi-chart layout, and a rule for total exposure. This is
the largest remaining design change and it is easy to do badly — a "portfolio"
that is really one position wearing three hats is worse than one market.

---

## 4. Agent-proposed market changes

**Status:** not built, and deliberately last.
**Trigger:** the operator has used the manual market switch for a while and has
a clear opinion about what the agent should be allowed to suggest.

The shape would be: the agent notices a market that fits a hypothesis, writes a
**proposal** with its reasoning and a proposed evaluation window, and the
operator approves it. Never automatic — choosing the market after seeing it win
is selection bias and would hollow out the track record.

---

## 5. Plan-mode execution for long research tasks

**Status:** not built.
**Trigger:** a research task appears that needs more than one model call to be
useful to the operator (e.g. "read this filing and tell me what changed").

The chat is one-shot by design. Multi-step work belongs in a separate, explicit
mode with its own budget and its own log, not smuggled into the chat box.

---

## 5b. A testnet order adapter

**Status:** not built. Reference: `docs/ORDER_API.md`.
**Trigger:** a strategy passes the gate, or the operator wants to rehearse
plumbing with fake money.

Order of work: `Broker` protocol → read-only reconcile → testnet orders with a
client order id on every request → one instrument at venue minimum with a hard
cap → only then reconsider the `live` tier. Python stdlib can do HMAC-SHA256 and
Ed25519 is available via `cryptography` if needed; Binance and Alpaca test hosts
exist so nothing has to touch real money to prove it works.

## 6. Live execution

**Status:** not built, by design. See `docs/PLATFORMS.md`.
**Trigger:** a strategy survives the gate **and** months of paper record **and**
a human raises the tier to `live` on a machine that has an adapter.

---

## How to use this file

When something is worth doing but not worth doing now, it goes here with a
trigger. When a trigger fires, the item moves into the roadmap with a date.
Items that never fire are deleted — a plan that only grows is a plan nobody
reads.
