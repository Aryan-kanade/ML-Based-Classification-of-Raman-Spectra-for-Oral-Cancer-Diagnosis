# Graph & Report Audit — full report (2026-09-12)

**Scope:** every graph in the project — all 32 GUI canvases and all 12
report artifacts — checked for correctness, self-labeling, axis
conventions, colors, guards, readability, and cross-report number
consistency, then **every finding fixed and re-verified**.

**Method:** real GUI rendered offscreen with the real dataset (317
spectra / 72 patients), real training (GNB + ET winner) and real
predictions; per-panel deterministic checks (title / axis labels /
inversion / legends via `axes_info`) plus PNG evidence captures in
`.zcode/graph_audit/`. A remote vision tool proved unreliable (returned
descriptions of the wrong images); the run logs are authoritative.

---

## Findings and fixes

### BUG-class

| # | Where | Problem | Fix | Verified |
|---|-------|---------|-----|----------|
| B1 | `save_result_figures` (gui.py) | "Save result figures" exported only 4 of the 6 Result-page charts — **calibration and decision-curve PNGs were silently missing** (they existed only inside the HTML report); docstring even said "three" | added `result_calibration.png` + `result_decision_curve.png` to the export set; docstring updated | re-audit: export list = 6 files; `test_freeze_and_figures_real_paths` now pins the set |
| B2 | `render_result_page` (gui.py) | switching to a winner **without** cm / out-of-fold probabilities left the **previous winner's confusion matrix and ROC on screen** (stale content presented as current) | else-branches clear both canvases to labelled placeholders — same pattern the calibration/DCA pair already used | re-audit: "19 stale-CM: cleared OK"; 0 findings |
| B3 | biochem panel (gui.py) | the **no-pairing fallback branch** (class means side by side) still labelled its y-axis "Δ marker (paired)" — wrong: those are not paired deltas | label is now conditional: "Δ marker (paired)" only when pairing exists, else "marker level (class mean)" | code-verified conditional; paired branch unchanged |
| B4 | `plot_prob_histogram` (plotting.py) | x-axis started at −0.02 — violated the project-wide zero-at-the-left-edge convention (§46) already applied to PR/ROC/calibration | `set_xlim(0, 1.02)` with the convention comment | pinned in test (`xlim[0] == 0.0`) |
| B5 | `plot_sign_bars` (plotting.py) | **dead code** — docstring claimed it was the shared look of the LOPO / biochem / NMF / local-explanation bars, but every one of those sites hand-rolls its own bars; only its pinning test ever called it | removed the function; pinning test rewritten to pin the *live* prob-histogram behaviour instead | `import plotting` OK; `test_plot_helpers_render` PASS |
| B6 | `plot_prob_histogram` (plotting.py) | y-ticks were one-per-count — a 299-spectra prediction set crammed **~300 y-ticks** (unreadable) | `MaxNLocator(integer=True)` — integer counts, auto density | pinned in test (≤12 ticks with n=299) |

### IMPROVE-class

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| I1 | Seed-stability boxplot | no x-axis label | "seed re-runs (5 points)" |
| I2 | LOPO per-patient bars | no x-axis label | "patient — hardest on the left" (bars are sorted worst-first) |

Both confirmed present in the re-audit's `axes_info` output.

### Reviewed and verified correct (no action)

- **Axis conventions** — every panel ascending; probability axes 0→1
  left-to-right, never inverted (§46/§47 checks logged per panel)
- **Confusion matrix** — counts + % of true class per cell, row labels
  carry (n=), colorbar, font scaling for dense matrices
- **ROC** — AUC in legend matches `roc_points` recomputation; Youden J
  operating point + chance line labelled; PR twin with AP
- **Calibration / DCA** — limits, identity line, raw vs calibrated
  series, treat-all/treat-none references
- **Preprocess preview** — twin-axis legends say which axis ("raw (left
  axis)" / "preprocessed (right axis)"); per-class means with n= in
  legends; paired card labelled with mode + patient counts
- **Colors** — consistent semantics everywhere: red = toward positive /
  weak (<0.5 LOPO) / result, blue = negative/away, amber = annotations
- **HTML report** — 6 embedded figures with consistent captions;
  honest-number policy with pending-warning; PPV-by-prevalence table;
  literature-context block
- **Cross-report numbers** — CM sums vs n, AUC legend vs computed,
  honest keys all consistent (txt / HTML / screen)
- Restored-winner CM (n=307) vs currently loaded data (n=299) in the
  audit environment is the documented display-only restore path, not a
  rendering bug

---

## Verification summary

| Check | Result |
|-------|--------|
| Result-page re-audit (exports, stale-CM, reports) | **0 findings** |
| Diagnostics re-audit (regions, seeds, noise, LOPO, LC, biochem, NMF) | **0 findings** |
| `test_plot_helpers_render` (pins B4/B6, B5 removal) | PASS |
| `test_freeze_and_figures_real_paths` (pins B1) | PASS |
| Full unit suite | 115/118 — the 3 failures (`test_prep_param_rows_and_compact`, `test_seq_results_html`, `test_stage3_conformal_permutation_explainability`) belong to a **parallel in-flight wave** (calculation-audit changes), not to any graph fix; they were failing before these fixes too |
| ruff (gui.py, plotting.py, test_all.py + app dir) | clean |
| Scientific invariance | display-only changes — LOPO macro-F1 0.586 and seed stability 0.576 ± 0.011 identical before vs after |

## Note on the parallel session

While this audit was being fixed, another active session was landing a
large uncommitted calculation-audit wave (≈1,500 lines across 15 files,
`SPIKE_REPORT.md`, `CALCULATION_AUDIT.md`, Brain.md §36/§37). The graph
fixes above were applied atomically against that moving tree and
coexist with it; nothing from that wave was committed or reverted by
this work.
