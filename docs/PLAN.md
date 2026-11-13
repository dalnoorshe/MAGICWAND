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
