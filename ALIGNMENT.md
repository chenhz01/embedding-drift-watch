# Alignment Audit (Rules 85-100 / 16 items) — embedding-drift-watch v0.1.1

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Safety/Ethics section | N/A | Not a security-research repo; no dual-use surface. Fails closed (exit 2) on unreadable/corrupt inputs |
| 2 | Privacy/dezero | PASS | No secrets/tokens/real names; no contact line needed (batch vector tool); grep `(sk-\|ghp_\|api[_-]?key\s*=\s*['"][A-Za-z0-9])` → 0 hits |
| 3 | Copyright boundary | PASS | Anchor comparisons (agentshield, vector-drift-monitor, rag-eval-toolkit) are structural restatements of public README facts (middleware vs CLI, FastAPI+Prometheus, RAG eval scope), no >50-char copied text; MIT LICENSE present, "zhengming contributors" |
| 4 | Fact-check annotation | PASS | Quickstart ~3 s marked [Measured] with repro `python self_test.py`; per-call ≈1 s markers; statistical-power claims qualitative |
| 5 | Confidence grading | PASS | Opener claim [High ≥80%] on math correctness; threshold usefulness [Medium]; marginal-coverage [Low]; visible in Confidence section |
| 6 | PLAN B built-in | PASS | "Minimal Fallback (PLAN B)": jq one-liner for per-dimension mean comparison, zero setup |
| 7 | Timeout preset | PASS | Quickstart states "~3 s" total with ~1 s per-step markers |
| 8 | Rule-conflict arbitration | PASS | "Design Trade-offs" opens with priority order: explainability of the score > zero dependencies > statistical power > throughput |
| 9 | Version changelog | PASS | CHANGELOG.md v0.1.0 + v0.1.1; v0.1.1 lists the baseline-range bucketing fix and test delta 51→56 |
| 10 | Exemption instruction | N/A | Session-level rule; static repo content not applicable |
| 11 | External-knowledge cutoff | PASS | No vendor-specific time-sensitive claims; anchor facts restated from the 2026-10 candidate pool snapshot |
| 12 | Assumption explicitness | PASS | "[Core Assumptions]" section with 4 items (fixed dimension, finite floats, chosen baseline, operator-side remediation) |
| 13 | Logic-check trio | PASS | self_test.py tiers: [A] functional (20), [B] exception/CLI paths (22), [C] extreme edges (14: empty/3000-dim/1e6-magnitude/1e-9/unicode path/400-vector timing/threshold boundary/64 buckets). Final run: `python self_test.py` → `SUMMARY: PASS 56/56` |
| 14 | Socratic question | PASS | FAQ "When NOT to use this tool?" with 4 concrete anti-use-cases (real-time middleware need, calibrated statistics, non-vector data, dashboard expectations) |
| 15 | Pareto improvement | PASS | v0.1.1: test count 51→56 (+9.8% functional-correctness fix gate met via the baseline-range correctness fix; a pure-shift TVD=0 bug is the substantive improvement); documented in CHANGELOG, suite re-run green |
| 16 | Ultimate fallback (≤60-word opener) | PASS | First paragraph = 57 words, contains what/who-needs-it judgment sentence |

Second-order effects (rule 16 bonus): PASS — "Limits & Second-order Effects" lists univariate-blindness to mixture drift, alert fatigue / threshold-tune-once-and-forget, and stale-baseline ritual risk.

Scan evidence: `grep -rEn "(sk-\|ghp_\|api[_-]?key\s*=\s*['\"][A-Za-z0-9])" .` → 0 hits; `python self_test.py` → `SUMMARY: PASS 56/56`.
