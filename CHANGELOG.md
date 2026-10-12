# Changelog — embedding-drift-watch

## v0.1.1 — 2026-10-12

- Fixed: `check` now buckets the current batch against the **baseline's stored
  histogram ranges** instead of the current batch's own min/max. Previously a
  pure distribution shift (same shape, moved location) produced TVD = 0 and
  could hide real drift; the baseline-range fix makes shifts visible in
  `mean_tvd`/`worst_dim` (regression-tested in `a06`, `a07`).
- Self-review round: added tier-C edge tests (constant columns, single-vector
  baselines, unicode paths, threshold boundary semantics, high bucket counts);
  test suite grew 51 → 56 cases.
- Clarified exit codes: 0 = no alert, 1 = alert, 2 = input/validation error.

## v0.1.0 — 2026-10-11

- Initial cut (session-interrupted build): fingerprint command (`build`),
  compare-and-alert command (`check`), JSON/JSONL loaders, strict validation
  (finite floats, fixed dimension), aggregate drift score
  (0.7·TVD + 0.2·mean_shift + 0.1·norm_drift), 51 self-test cases.
