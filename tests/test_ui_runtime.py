"""Execute the console's JavaScript against a stub DOM.

`node --check` only validates syntax; it cannot see an undefined variable. That
is how "declined is not defined" shipped: the declaration's indentation did not
match, the replace silently did nothing, and the whole refresh loop threw at that
panel — freezing autopilot, audit, storage and the feed while the header still
updated, which looked like "nothing is working".

These tests run the real functions with realistic payloads.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "magicwand" / "dashboard.html"


HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const js = src.match(/<script>([\s\S]*?)<\/script>/)[1];

function el() {
  return {
    innerHTML: "", textContent: "", className: "", value: "", checked: false,
    title: "", src: "", style: {}, scrollTop: 0, scrollHeight: 0, dataset: {},
    classList: {add(){}, remove(){}, contains(){return false;}, toggle(){}},
    appendChild(){}, insertAdjacentHTML(){}, addEventListener(){},
    setAttribute(){}, removeAttribute(){}, focus(){}, querySelectorAll(){return [];},
    onclick: null, onchange: null,
  };
}
const cache = {};
global.document = {
  getElementById: (id) => (cache[id] = cache[id] || el()),
  addEventListener(){},
  querySelectorAll(){ return []; },
};
global.window = {};
global.setInterval = () => 0;
global.setTimeout = (f) => { try { f(); } catch (e) {} return 0; };
global.clearTimeout = () => {};
global.confirm = () => false;
global.alert = () => {};
global.TextDecoder = class { decode(v){ return String(v); } };
global.fetch = async () => ({
  ok: true, status: 200,
  json: async () => ({ messages: [], models: {models: []}, result: {}, ok: true,
                       settings: {}, state: {} }),
  text: async () => "",
  body: null,
});

const state = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

try {
  eval(js);
} catch (e) {
  console.log("EVAL FAILED: " + e.message);
  process.exit(1);
}

const research = {
  ts: "2026-11-16T07:00:00Z", decision: "veto", evidence_score: -0.42,
  summary: "5/5 readers reported", stub: false,
  single_source_declined: ["news"],
  agents: [
    {agent: "news", stance: "veto", confidence: 0.68, evidence_score: -0.55,
     corroboration: 1, thesis: "risk off", error: ""},
    {agent: "social", stance: "neutral", confidence: 0.5, evidence_score: 0.1,
     corroboration: 3, thesis: "quiet", error: ""},
    {agent: "onchain", stance: "neutral", confidence: 0.0, evidence_score: 0.0,
     corroboration: 0, thesis: "", error: "HTTP 500"},
  ],
  chart: {stance: "neutral", confidence: 0.5, thesis: "range",
          observations: ["a"], injected_text_seen: true, error: ""},
};

const calls = [
  ["renderAgents", () => renderAgents(research)],
  ["renderAgents stub", () => renderAgents({stub: true, agents: []})],
  ["renderAgents empty", () => renderAgents(null)],
  ["renderAgents aged", () => renderAgents(research, {server_time: "2026-11-16T10:00:00Z",
                                                       autopilot: {interval: 900}})],
  ["renderAgents inflight", () => renderAgents(research, {server_time: "2026-11-16T07:03:00Z",
                                                          research_inflight: {age_seconds: 12}})],
  ["renderDesk", () => renderDesk(state.desk)],
  ["renderAudit", () => renderAudit(state.audit)],
  ["renderAudit empty", () => renderAudit({})],
  ["renderProposals", () => renderProposals(state.proposals)],
  ["renderReflection", () => renderReflection(state.reflection, state.memory)],
  ["renderStrategies", () => renderStrategies(state.strategies)],
  ["renderFeed", () => renderFeed(state.recent)],
  ["renderAutopilot", () => renderAutopilot(state.autopilot)],
  ["renderChart", () => renderChart(state)],
  ["renderTiers", () => renderTiers(state.tiers, state.policy.tier)],
  ["md", () => md("**bold** `code`\n- a\n- b\n# head")],
  ["renderChat", () => renderChat([{role:"user", text:"hi", ts:"t"},
                                   {role:"assistant", text:"yo", ts:"t",
                                    sources:[{title:"T", url:"https://a"}]}])],
];

let failed = 0;
for (const [name, fn] of calls) {
  try { fn(); }
  catch (e) { failed++; console.log("RENDER FAILED [" + name + "]: " + e.message); }
}
// Freshness has to be stated, not implied: a four-hour-old reading and a
// one-second-old one rendered identically, which is why the panel felt faked.
function check(name, cond, detail) {
  if (!cond) { failed++; console.log("ASSERT FAILED [" + name + "]: " + detail); }
}
const staleState = {server_time: "2026-11-16T10:00:00Z", autopilot: {interval: 900}};
renderAgents(research, staleState);
const tsEl = document.getElementById("researchTs");
check("age is shown", /researched.*ago/.test(tsEl.textContent), tsEl.textContent);
check("age has an absolute title", !!tsEl.title, "empty title");
const staleHtml = document.getElementById("agents").innerHTML;
check("stale reading is flagged", /stale reading/i.test(staleHtml), staleHtml.slice(0, 120));
check("stale note names the time", /recorded, at/i.test(staleHtml), staleHtml.slice(0, 200));

renderAgents(research, {server_time: "2026-11-16T07:03:00Z", autopilot: {interval: 900}});
check("fresh reading has no stale banner",
      !/stale reading/i.test(document.getElementById("agents").innerHTML), "banner on fresh");
check("fresh age reads in minutes",
      /researched 3 min ago/.test(document.getElementById("researchTs").textContent),
      document.getElementById("researchTs").textContent);

renderAgents(research, {server_time: "2026-11-16T07:03:00Z",
                        research_inflight: {age_seconds: 12}});
// A running cycle must paint itself: the reader that came back, the one being
// read right now, and the ones still queued — never the previous cycle's
// verdicts dressed up as this one's.
renderAgents(research, {server_time: "2026-11-16T07:05:00Z",
                        research_inflight: {
                          started_at: "2026-11-16T07:04:44Z", age_seconds: 16,
                          phase: "reading", current: "social",
                          plan: ["news", "social", "onchain"],
                          done: [{agent: "news", stance: "veto", confidence: 0.62,
                                  evidence_score: -0.45, corroboration: 2,
                                  thesis: "risk off now", error: ""}],
                          failing: []}});
const liveHtml = document.getElementById("agents").innerHTML;
const liveDec = document.getElementById("decision").innerHTML;
check("live view shows the returned reader", /risk off now/.test(liveHtml), liveHtml.slice(0, 200));
check("live view shows the reader in flight", /reading now/.test(liveHtml), liveHtml.slice(0, 400));
check("live view shows queued readers", /queued/.test(liveHtml), liveHtml.slice(0, 400));
check("live view counts readers in", /1 of 3 readers in/.test(liveDec), liveDec);
check("live view shows elapsed time", /0:1[6-9]/.test(liveDec), liveDec);
check("live view keeps the last verdict visible", /last complete cycle/.test(liveDec), liveDec);
check("live view draws no bar for a queued reader",
      (liveHtml.match(/width:0%/g) || []).length === 1, liveHtml.slice(0, 400));
check("a running cycle is visible",
      /cycle running|reading the market/i.test(document.getElementById("decision").innerHTML),
      document.getElementById("decision").innerHTML);
check("a running cycle suppresses the stale banner",
      !/stale reading/i.test(document.getElementById("agents").innerHTML), "banner during run");
check("a running cycle says so in the header",
      /new cycle running/.test(document.getElementById("researchTs").textContent),
      document.getElementById("researchTs").textContent);

if (failed) process.exit(1);
console.log("all panels rendered: " + calls.length);
"""

STATE = {
    "server_time": "t", "symbol": "SOL-USD", "exchange": "coinbase",
    "timeframe": "1h", "account_equity": 50.0,
    "account": {"equity": 50.0, "source": "config", "detail": "", "stale": False},
    "policy": {"tier": 0, "tier_name": "research", "effective": {}, "reasoning": ""},
    "tiers": {"0": "research", "1": "paper_desk"},
    "desk": {"cash": 5.0, "position": None, "trades": 0, "win_rate": None,
             "autonomy_armed": True, "revoked_reason": "", "trades_today": 0,
             "recent_trades": []},
    "research": {"decision": "veto", "agents": [], "summary": "s"},
    "chart": {"file": "c.png", "kb": 20.0, "at": "2026-11-16T07:00:00Z"},
    "reflection": {"ts": "t", "changes": ["shrank risk"], "summary": "s",
                   "sample": {"trades": 1, "graded_agents": 2}},
    "memory": {"blacklisted_families": ["sma_trend"], "preferred_families": [],
               "agent_weights": {}},
    "strategies": [{"name": "s", "family": "breakout", "metrics": {},
                    "deployed": False}],
    "proposals": {"total": 10, "validated": 3, "counts": {"passed": 0},
                  "failing_gates": {"profitable": 3},
                  "recent": [{"name": "n", "family": "f", "ok": False,
                              "summary": "rejected", "metrics": {}}]},
    "ledger": {"ok": True, "message": "chain intact", "entries": 10},
    "equity_series": [{"ts": "t", "equity": 50.0}, {"ts": "t2", "equity": 51.0}],
    "recent": [{"kind": "mark", "ts": "t", "payload": {"equity": 50}}],
    "autopilot": {"running": True, "ticks": 3, "interval": 900, "errors": 0,
                  "estimated_daily_cost_usd": 0.38, "next_in_seconds": 10,
                  "last_result": "ok"},
    "audit": {"ts": "2026-11-16T07:00:00Z", "gaps": 0, "passes": 23, "notes": 4},
    "backups": {"archives": 4, "age_hours": 2.6, "enabled": True,
                "interval_hours": 12, "keep": 14},
    "settings": {"has_key": True, "model": "deepseek-flash",
                 "markets": [{"symbol": "SOL-USD", "name": "Solana"}],
                 "symbol": "SOL-USD", "exchange": "coinbase"},
    "models": {"models": []}, "service": {}, "model_mode": "live",
}


@unittest.skipUnless(shutil.which("node"), "node is required for the UI runtime test")
class TestConsoleRuntime(unittest.TestCase):
    def test_every_panel_renders_without_throwing(self):
        with tempfile.TemporaryDirectory() as d:
            harness = Path(d) / "h.js"
            harness.write_text(HARNESS)
            state = Path(d) / "state.json"
            state.write_text(json.dumps(STATE))
            r = subprocess.run(["node", str(harness), str(PAGE), str(state)],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("all panels rendered", r.stdout)

    def test_declined_is_declared_before_use(self):
        """The exact bug: used in renderAgents, declared nowhere."""
        html = PAGE.read_text()
        fn = html[html.index("function renderAgents"):]
        fn = fn[:fn.index("\nfunction ", 10)]
        self.assertIn("const declined", fn,
                      "single_source_declined must be declared in renderAgents")
        self.assertLess(fn.index("const declined"),
                        fn.index("declined.length"))

    def test_every_panel_call_is_guarded(self):
        html = PAGE.read_text()
        for name in ("agents", "desk", "audit", "storage", "autopilot",
                     "proposals", "strategies", "feed"):
            self.assertIn(f'safe("{name}"', html, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
