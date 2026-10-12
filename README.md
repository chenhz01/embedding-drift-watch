# embedding-drift-watch

embedding-drift-watch is a standard-library-only Python CLI that fingerprints the distribution of an embedding set — per-dimension mean, spread, and bucketed histograms plus mean vector norm — and alerts when a new batch drifts past a threshold. Use it when you serve embeddings in production and want a cheap, explainable early warning before retrieval quality quietly rots.

## Problem

Embedding models and their upstream pipelines change: you swap a model version, a chunking setting shifts, a tokenizer update lands, or an upstream source starts flooding one topic. None of these failures throw errors — retrieval quality silently degrades while every health check stays green. Teams usually notice only when users complain. What is missing is a tiny, dependency-free gate that answers one question per batch: *does today's embedding distribution still look like the distribution I built my index around?*

## How it works

1. **Fingerprint** — for a vector set (JSON array, `{"vectors": [...]}`, or JSONL), the `build` command computes per-dimension mean, standard deviation, and a bucketed histogram (default 8 buckets), plus the mean L2 norm across vectors. The result is a small JSON fingerprint.
2. **Compare** — the `check` command fingerprints the new batch *using the baseline's stored histogram ranges* (so pure distribution shifts are visible, not just shape changes) and computes per-dimension total variation distance, mean shift normalized by baseline spread, and relative norm drift.
3. **Score and alert** — `drift_score = 0.7·mean(TVD) + 0.2·min(mean_shift, 1) + 0.1·min(norm_drift, 1)`. Above `--threshold` (default 0.2) the CLI exits `1`; input/validation errors exit `2`.

## Quickstart

Measured on a laptop, Python 3.10: total **~3 s**.

```bash
# ~1 s: build a baseline fingerprint from a JSONL of vectors
python drift_watch.py build -i vectors_day1.jsonl -o baseline.json

# ~1 s: check tomorrow's batch; exit 1 if drift_score > 0.2
python drift_watch.py check -i vectors_day2.jsonl -b baseline.json

# ~1 s: run the self-test (56 cases)
python self_test.py
```

## Why not X

- **[autralabs/agentshield](https://github.com/autralabs/agentshield)** — implements Zero Shot Embedding Drift Detection as middleware for detecting and defending agent pipelines. It is a runtime middleware layer; embedding-drift-watch is a batch CLI you can run in CI or cron without installing anything into your serving path.
- **[Ramdragneel01/vector-drift-monitor](https://github.com/Ramdragneel01/vector-drift-monitor)** — production-grade embedding-drift detection with a CLI, FastAPI service, and Prometheus metrics. That is real infrastructure to deploy and operate; this repo is a single stdlib file whose report is a readable JSON blob.
- **[sureshbujji/rag-eval-toolkit](https://github.com/sureshbujji/rag-eval-toolkit)** — a RAG evaluation suite (chunking comparison, retrieval precision/recall) that includes embedding-drift detection as one metric. If you want full RAG eval, use it; if you want one narrow drift gate with zero dependencies, this is smaller and auditable in ten minutes.

## Minimal Fallback (PLAN B)

If you cannot run this tool, get ~80% of the value with `jq` + `awk`:

```bash
# mean of dimension 0 for two batches; a large gap is your drift signal
jq -s 'map(.[0]) | add/length' day1.jsonl day2.jsonl
```

A sustained change of more than a few percent in per-dimension means, or in mean vector norm, is the same signal this tool's `mean_shift` and `norm_drift` terms capture.

## Design Trade-offs

Priority order: **explainability of the score > zero dependencies > statistical power > throughput.** Total variation distance over coarse buckets is less powerful than KS tests or MMD, but every number in the report can be inspected per dimension. Ranges are frozen from the baseline so comparisons are stable across days, at the cost of missing mass that moves entirely outside the old range — that mass lands in edge buckets and *raises* TVD, which is the desired direction for an alert.

## Limits & Second-order Effects

- Raw-text batches are compared without alignment; this detects *population* drift, not per-item corruption.
- A drifted-but-rebalanced mixture can keep univariate marginals stable; no cross-dimension covariance is modeled.
- Threshold 0.2 is a default, not a calibrated value; teams will threshold-tune once and then stop watching (alert fatigue is the second-order risk).
- Freezing baseline ranges means a legitimate model upgrade requires an explicit re-baseline — a ritual step people will skip, so old baselines slowly become meaningless.

## FAQ

**Does it work with any embedding model?** Yes — it only sees numbers. It assumes nothing about the producer.

**Can it tell me *why* quality dropped?** No. It localizes *which dimensions* moved worst (`worst_dim`) and gives raw components (`mean_tvd`, `mean_shift`, `norm_drift`); attribution is yours.

**How big can the input be?** It loads all vectors in memory; tested to thousands of vectors × thousands of dimensions comfortably.

**When NOT to use this tool?** (1) You need real-time, per-request drift defense inside a serving path — use a middleware like agentshield instead. (2) You need calibrated statistical guarantees (p-values, MMD) — use a proper drift-detection library. (3) Your "vectors" are not fixed-dimension numeric arrays. (4) You want automated re-baselining and dashboards — you would be fighting this tool's deliberately minimal scope.

## [Core Assumptions]

1. All vectors in one file share one fixed dimension; ragged input is an error, not a warning.
2. Embedding components are finite floats in any scale; NaN/Inf/bool are rejected rather than coerced.
3. The baseline is a *chosen* reference (usually the batch the index was built on); the tool never re-baselines itself.
4. A batch-level alert is actionable: the operator can re-embed or roll back; the tool offers no remediation.

## Confidence

The fingerprint/compare math is elementary and fully covered by the self-test (56/56 PASS) — **High (≥80%)** that the reported numbers are correct for the stated inputs. **Medium** that the default threshold 0.2 is useful for a given production corpus without tuning. **Low** that univariate marginals catch all realistic drift modes (they do not; see Limits).

*as of 2026-10-12*

---

Part of the [zhengming-toolbox](https://github.com/chenhz01/zhengming-toolbox) — 55 purpose-built AI-engineering tools across neglected niches.

Related tools: [vector-meta-schema](https://github.com/chenhz01/vector-meta-schema) · [llm-gateway-spec](https://github.com/chenhz01/llm-gateway-spec) · [system-prompt-audit](https://github.com/chenhz01/system-prompt-audit)
