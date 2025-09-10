
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from numerical_equilibrium_solver_v4 import (Params, asdict, rebase_L_for_theta_one,
    sweep_DW_piecewise, T_of_X, solve)

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

p0 = rebase_L_for_theta_one(Params())

# DW sweep
DW_grid = list(np.round(np.concatenate([
    np.linspace(0.10, 0.20, 7),
    np.linspace(0.205, 0.235, 7),
    np.linspace(0.240, 0.255, 7),
    np.linspace(0.256, 0.34, 12)
]), 3))

results_DW, DW_star = sweep_DW_piecewise(p0, DW_grid, enforce_monotone=True)
rows = []
for D, e in zip(DW_grid, results_DW):
    rows.append({"D_W": D, "x_D": e.x_D, "x_W": e.x_W, "X": e.X, "r_w": 0.0 if e.r_w is None else e.r_w,
                 "A_D_star": e.A_D_star, "A_W_star": (float("nan") if e.A_W_star is None else e.A_W_star),
                 "A_W_starstar": (float("nan") if e.A_W_starstar is None else e.A_W_starstar),
                 "pD": e.pD, "pW": e.pW, "regime": e.regime})
dw_df = pd.DataFrame(rows)
dw_df.to_csv(os.path.join(OUT, "results_DW_sweep.csv"), index=False)

# τ sweeps
def tau_sweep_at(DW_value, taus):
    rows = []; seed=None
    for t in taus:
        p = rebase_L_for_theta_one(Params()); p.D_W = DW_value; p.tau = float(t)
        eq = solve(p, seed); seed=(eq.x_D,eq.x_W)
        rows.append({"tau": float(t), "x_D": eq.x_D, "x_W": eq.x_W, "X": eq.X,
                     "r_w": (0.0 if eq.r_w is None else eq.r_w), "pD": eq.pD, "pW": eq.pW, "regime": eq.regime})
    return pd.DataFrame(rows)

taus = np.round(np.linspace(0.0, 0.010, 11), 3)
tau_mixed    = tau_sweep_at(0.12, taus)
tau_boundary = tau_sweep_at(0.235, taus)
tau_slack    = tau_sweep_at(0.30, taus)

tau_mixed.to_csv(os.path.join(OUT, "results_tau_mixed.csv"), index=False)
tau_boundary.to_csv(os.path.join(OUT, "results_tau_boundary.csv"), index=False)
tau_slack.to_csv(os.path.join(OUT, "results_tau_slack.csv"), index=False)

print("Done. CSVs written to", OUT)
