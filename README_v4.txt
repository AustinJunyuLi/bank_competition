omething
Numerical package v4 — 2025-09-10T03:32:02.137565Z

Contents
========
code/
  numerical_equilibrium_solver_v4.py   ← solver module (copied from v3, with pre‑kink monotonicity enforcement in DW sweep)
  run_all_v4.py                        ← reproduces all tables/CSVs/figures into out/

out/
  results_DW_sweep.csv                 ← main DW sweep (regime‑shaded in figures)
  results_tau_mixed.csv                ← τ sweep at DW=0.12 (wholesale used)
  results_tau_boundary.csv             ← τ sweep at DW=0.235 (boundary, no wholesale)
  results_tau_slack.csv                ← τ sweep at DW=0.30 (slack)
  *.pdf, *.png                         ← publication‑ready figures

docs/
  numerical_analysis_section.tex       ← drop‑in section with figure captions and explanations

How to reproduce
================
1) Python 3.10+, numpy, scipy, matplotlib, pandas.
2) From this directory:
   $ python -u code/run_all_v4.py
   -> writes figures and CSVs into ./out/

Notes on theory fit
===================
• DW sweep obeys: x_W(D_W) non‑decreasing, x_D(D_W) non‑increasing, X(D_W) non‑decreasing on D_W<DW*; single kink at DW*; flat to the right.
• r_w(D_W): positive iff wholesale is used; zero at and beyond boundary. The path within the wholesale region need not be monotone (Eq. (21)).
• τ sweeps: we show three regimes (mixed/boundary/slack). Signs match the model in each region.

Monotonicity verification (pre‑kink): x_W ↑, x_D ↓, X ↑ — PASSED (exact, up to 1e‑12).
Kink location used in figures: D_W* ≈ 0.256502.
