
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import numpy as np
import math

# ================== Parameters ==================

@dataclass
class Params:
    alpha: float = 0.3
    pi: float = 0.95
    L: float = 1.0
    gamma: float = 0.10
    tau: float = 0.003
    D_W: float = 0.45
    A_low: float = 0.5
    A_high: float = 1.5

    # numerics
    tol: float = 1e-10
    max_iter: int = 250
    ls_shrink: float = 0.5
    ls_min: float = 1e-7
    ridge: float = 2e-8
    fd_eps: float = 1e-6
    cap_deadband: float = 1e-10

    def theta(self) -> float:
        return self.pi * (self.alpha ** 2) * (self.L ** (1.0 - self.alpha))

    def invpow(self) -> float:
        return self.alpha - 1.0  # 1/ε ∈ (-1,0)

    def check(self) -> None:
        assert 0.0 < self.alpha < 1.0
        assert 0.0 < self.pi < 1.0
        assert 0.0 < self.L
        assert 0.0 < self.gamma < 1.0
        assert self.A_low > self.alpha, "Need A_low > alpha to preclude strategic default."
        assert self.A_low < 1.0 < self.A_high

@dataclass
class Eq:
    regime: str
    x_D: float
    x_W: float
    r_w: Optional[float]
    X: float
    T: float
    A_D_star: float
    A_W_star: Optional[float]
    A_W_starstar: Optional[float]
    pD: float
    pW: float
    iters: int
    converged: bool
    notes: str

def clip(z: float, a: float, b: float) -> float:
    return max(a, min(b, z))

def safe_pow(x: float, p: float, eps: float = 1e-14) -> float:
    return (max(x, eps)) ** p

def T_of_X(p: Params, X: float) -> float:
    return p.theta() * safe_pow(X, p.invpow())

def marginal_term(p: Params, X: float, x_self: float, x_other: float) -> float:
    return p.theta() * safe_pow(X, p.invpow() - 1.0) * (x_other + p.alpha * x_self)

def thresholds_slack(p: Params, x_D: float, x_W: float) -> Tuple[float, float]:
    X = x_D + x_W
    T = T_of_X(p, X)
    A_star = (1.0 - p.gamma) / T
    return A_star, A_star

def thresholds_mixed(p: Params, x_D: float, x_W: float, r_w: float) -> Tuple[float, float, float]:
    X = x_D + x_W
    T = T_of_X(p, X)
    A_D = (1.0 - p.gamma) / T
    A_W = ((1.0 + r_w) * (1.0 - p.gamma) - r_w * p.D_W / max(x_W, 1e-16)) / T
    A_Wss = p.D_W / (max(x_W, 1e-16) * T)
    return A_D, A_W, A_Wss

# Uniform distribution integrals
def tail_prob_uniform(z: float, a: float, b: float) -> float:
    if z <= a: return 1.0
    if z >= b: return 0.0
    return (b - z) / (b - a)

def int_1_uniform(a: float, b: float, z: float) -> float:
    return max(0.0, min(z, b) - a)

def int_A_uniform(a: float, b: float, z: float) -> float:
    zc = clip(z, a, b)
    return max(0.0, 0.5 * (zc*zc - a*a))

# Deposit subsidy for D under Uniform
def subsidy_D(p: Params, x_D: float, x_W: float) -> float:
    a, b = p.A_low, p.A_high
    X = x_D + x_W
    T = T_of_X(p, X)
    z = clip((1.0 - p.gamma)/T, a, b)
    M = marginal_term(p, X, x_D, x_W)
    num = (1.0 - p.gamma) * int_1_uniform(a, b, z) - M * int_A_uniform(a, b, z)
    return num / (b - a)

# Wholesale zero-profit Φ and partials (analytic Φ_r; numeric Φ_xW)
def phi_value_and_partials(p: Params, x_D: float, x_W: float, r_w: float):
    a, b = p.A_low, p.A_high
    f0 = 1.0 / (b - a)
    X = x_D + x_W
    T = T_of_X(p, X)
    W = (1.0 - p.gamma) * x_W - p.D_W
    if W <= 0.0:
        return 0.0, 0.0, 0.0

    A_W = ((1.0 + r_w) * (1.0 - p.gamma) - r_w * p.D_W / max(x_W, 1e-16)) / T
    A_Wss = p.D_W / (max(x_W, 1e-16) * T)

    tail = tail_prob_uniform(A_W, a, b)
    z1 = clip(A_Wss, a, b)
    z2 = clip(A_W, a, b)
    resid = 0.0
    if z2 > z1:
        resid = (x_W * T * 0.5 * (z2*z2 - z1*z1) - p.D_W * (z2 - z1)) * f0

    Phi = (1.0 + r_w) * W * tail + resid - W

    # analytic Φ_r
    if A_W <= a:
        Phi_r = W
    elif A_W >= b:
        Phi_r = 0.0
    else:
        dA_dr = W / (x_W * T)
        Phi_r = W * tail + (1.0 + r_w) * W * (-f0) * dA_dr + f0 * (x_W * T) * (A_W - A_Wss) * dA_dr

    # numeric Φ_xW (keep r fixed)
    he = p.fd_eps * (1.0 + abs(x_W))
    xWm = max(1e-12, x_W - he)
    xWp = x_W + he
    def phi_at(xWv: float) -> float:
        Xv = x_D + xWv
        Tv = T_of_X(p, Xv)
        Wv = (1.0 - p.gamma) * xWv - p.D_W
        if Wv <= 0.0:
            return 0.0
        A_Wv = ((1.0 + r_w) * (1.0 - p.gamma) - r_w * p.D_W / max(xWv, 1e-16)) / Tv
        A_Wssv = p.D_W / (max(xWv, 1e-16) * Tv)
        tailv = tail_prob_uniform(A_Wv, a, b)
        z1v = clip(A_Wssv, a, b); z2v = clip(A_Wv, a, b)
        residv = 0.0
        if z2v > z1v:
            residv = (xWv * Tv * 0.5 * (z2v*z2v - z1v*z1v) - p.D_W * (z2v - z1v)) * f0
        return (1.0 + r_w) * Wv * tailv + residv - Wv
    Phi_xW = (phi_at(xWp) - phi_at(xWm)) / (xWp - xWm)

    return Phi, Phi_r, Phi_xW

# FOCs
def FD_slack(p: Params, x_D: float, x_W: float) -> float:
    X = x_D + x_W
    return (marginal_term(p, X, x_D, x_W) - 1.0
            - p.tau * (1.0 - p.gamma)
            + subsidy_D(p, x_D, x_W))

def FW_partial_mixed(p: Params, x_D: float, x_W: float, r_w: float) -> float:
    X = x_D + x_W
    MR = p.theta() * (max(X,1e-14) ** (p.invpow()-1.0)) * (x_D + p.alpha * x_W)
    a, b = p.A_low, p.A_high
    T = T_of_X(p, X)
    A_w_ss = p.D_W / (max(x_W, 1e-16) * T)
    z = clip(A_w_ss, a, b)
    intA = int_A_uniform(a, b, z) / (b - a)
    neg_integral = - MR * intA
    return MR - 1.0 - r_w * (1.0 - p.gamma) + neg_integral

def dwdrr_partial_mixed(p: Params, x_D: float, x_W: float, r_w: float) -> float:
    W = (1.0 - p.gamma) * x_W - p.D_W
    return -W if W > 0.0 else 0.0

# Numerics


def wW_value(p: Params, x_D: float, x_W: float) -> float:
    """Bank W objective value at (x_D, x_W), solving r_w endogenously if needed,
    and using the correct integral forms under Uniform[A_low, A_high]."""
    X = x_D + x_W
    T = T_of_X(p, X)
    a, b = p.A_low, p.A_high
    f0 = 1.0/(b - a)
    # Wholesale amount (if any)
    W = (1.0 - p.gamma) * x_W - p.D_W
    private = x_W * (T) - x_W  # x_W*(Θ X^{1/ε} - 1)
    if W <= 0.0:
        # boundary: no wholesale; subsidy integral over [a, A_Wss]
        A_Wss = p.D_W/(max(x_W,1e-16)*T)
        z = clip(A_Wss, a, b)
        intA = int_A_uniform(a, b, z)*f0
        subsidy = p.D_W * tail_prob_uniform(A_Wss, a, b) - x_W*T*intA  # E[S_W] at boundary
        return private + subsidy - p.tau * p.D_W
    else:
        # Mixed: solve r_w and compute full expression
        r_w = solve_rw_bisect(p, x_D, x_W)
        # Subsidy to depositors only occurs for A < A_Wss
        A_Wss = p.D_W/(max(x_W,1e-16)*T)
        z = clip(A_Wss, a, b)
        intA = int_A_uniform(a,b,z)*f0
        subsidy = p.D_W * tail_prob_uniform(A_Wss, a, b) - x_W*T*intA
        # Wholesale cost term
        cost_wholesale = r_w * W
        return private - cost_wholesale + subsidy - p.tau * p.D_W

def FW_total_marginal(p: Params, x_D: float, x_W: float) -> float:
    """Compute d/dx_W of wW_value with r_w endogenously solved (total derivative)."""
    h = p.fd_eps * (1.0 + abs(x_W))
    xm = max(1e-10, x_W - h)
    xp = x_W + h
    wm = wW_value(p, x_D, xm)
    wp = wW_value(p, x_D, xp)
    return (wp - wm) / (xp - xm)


def finite_diff(f, x: np.ndarray, h: float) -> np.ndarray:
    fx = f(x)
    m, n = len(fx), len(x)
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        step = h * (1.0 + abs(x[j]))
        xp = x.copy(); xp[j] += step
        J[:, j] = (f(xp) - fx) / step
    return J

def damped_newton(f, x0: np.ndarray, tol: float, max_iter: int, ridge: float, shrink: float, min_step: float, fd_eps: float):
    x = x0.copy()
    fx = f(x)
    for it in range(1, max_iter+1):
        J = finite_diff(f, x, fd_eps)
        JTJ = J.T @ J
        g = J.T @ fx
        lam = ridge * (1.0 + float(np.linalg.norm(g)))
        try:
            s = -np.linalg.solve(JTJ + lam*np.eye(JTJ.shape[0]), g)
        except np.linalg.LinAlgError:
            s = -np.linalg.lstsq(J, fx, rcond=None)[0]
        t = 1.0; f0 = float(np.linalg.norm(fx)); improved = False
        while t >= min_step:
            xn = x + t * s
            fn = f(xn)
            if float(np.linalg.norm(fn)) < f0:
                x, fx = xn, fn
                improved = True
                break
            t *= shrink
        if not improved:
            return x, False, it
        if float(np.linalg.norm(fx)) < tol:
            return x, True, it
    return x, False, max_iter

# Solvers
def solve_slack(p: Params, xD0: float, xW0: float) -> Tuple[float, float, bool, int]:
    def F(z: np.ndarray) -> np.ndarray:
        xD, xW = float(z[0]), float(z[1])
        return np.array([FD_slack(p, xD, xW),
                         FD_slack(p, xW, xD)], dtype=float)
    root, ok, it = damped_newton(F, np.array([xD0, xW0], dtype=float),
                                 p.tol, p.max_iter, p.ridge, p.ls_shrink, p.ls_min, p.fd_eps)
    return float(root[0]), float(root[1]), ok, it

def solve_rw_bisect(p: Params, x_D: float, x_W: float) -> float:
    W = (1.0 - p.gamma) * x_W - p.D_W
    if W <= 0.0:
        return 0.0
    lo, hi = 0.0, 10.0
    def phi_r(r):
        return phi_value_and_partials(p, x_D, x_W, r)[0]
    flo = phi_r(lo)
    it = 0
    fhi = phi_r(hi)
    while (np.sign(fhi) == np.sign(flo)) and it < 60:
        hi *= 2.0; it += 1; fhi = phi_r(hi)
    if np.sign(fhi) == np.sign(flo):
        return lo if abs(flo) <= abs(fhi) else hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = phi_r(mid)
        if abs(fm) < 1e-12: return mid
        if np.sign(fm) == np.sign(flo):
            lo = mid; flo = fm
        else:
            hi = mid; fhi = fm
    return 0.5 * (lo + hi)

def solve_mixed_nested(p: Params, xD0: float, xW0: float) -> Tuple[float, float, float, bool, int]:
    def F(z: np.ndarray) -> np.ndarray:
        xD, xW = float(z[0]), float(z[1])
        min_xW = p.D_W / max(1e-16,(1.0 - p.gamma))
        if xW < min_xW:
            xW = min_xW
        rw = solve_rw_bisect(p, xD, xW)
        Phi, Phi_r, Phi_xW = phi_value_and_partials(p, xD, xW, rw)
        drdxW = 0.0
        if abs(Phi_r) > 1e-14:
            drdxW = - Phi_xW / Phi_r
        f1 = FD_slack(p, xD, xW)  # D FOC structure (premium marginal when slack)
        f2 = FW_partial_mixed(p, xD, xW, rw) + dwdrr_partial_mixed(p, xD, xW, rw) * drdxW
        return np.array([f1, f2], dtype=float)
    root, ok, it = damped_newton(F, np.array([xD0, xW0], dtype=float),
                                 p.tol, p.max_iter, p.ridge, p.ls_shrink, p.ls_min, p.fd_eps)
    xD, xW = float(root[0]), float(root[1])
    min_xW = p.D_W / max(1e-16,(1.0 - p.gamma))
    if xW < min_xW:
        xW = min_xW
    rw = solve_rw_bisect(p, xD, xW)
    return xD, xW, rw, ok, it

def solve(p: Params, x_guess: Optional[Tuple[float, float]] = None) -> Eq:
    p.check()
    invpow = p.invpow()
    if x_guess is None:
        X0 = max(1e-6, (1.0 / p.theta()) ** (1.0 / invpow))
        xD0 = 0.5 * X0; xW0 = 0.5 * X0
    else:
        xD0, xW0 = x_guess

    # Try slack
    xDs, xWs, ok_s, it_s = solve_slack(p, xD0, xW0)
    gap = (1.0 - p.gamma) * xWs - p.D_W
    if ok_s and gap < -p.cap_deadband:
        X = xDs + xWs; T = T_of_X(p, X)
        Astar = (1.0 - p.gamma) / T
        def cdf(z: float) -> float:
            if z <= p.A_low: return 0.0
            if z >= p.A_high: return 1.0
            return (z - p.A_low) / (p.A_high - p.A_low)
        pd = cdf(Astar)
        return Eq("slack", xDs, xWs, None, X, T, Astar, None, None, pd, pd, it_s, True, "Slack-cap equilibrium.")

    # Mixed
    min_xW = p.D_W / max(1e-16,(1.0 - p.gamma))
    xD0m = max(xDs if ok_s else xD0, 1e-8)
    xW0m = max(xWs if ok_s else xW0, min_xW)
    xDm, xWm, rwm, ok_m, it_m = solve_mixed_nested(p, xD0m, xW0m)
    gap_m = (1.0 - p.gamma) * xWm - p.D_W
    if gap_m <= -p.cap_deadband:
        # fallback to slack
        xDs2, xWs2, ok_s2, it_s2 = solve_slack(p, xD0, xW0)
        X2 = xDs2 + xWs2; T2 = T_of_X(p, X2)
        Astar2 = (1.0 - p.gamma) / T2
        def cdf2(z: float) -> float:
            if z <= p.A_low: return 0.0
            if z >= p.A_high: return 1.0
            return (z - p.A_low) / (p.A_high - p.A_low)
        pd2 = cdf2(Astar2)
        return Eq("slack", xDs2, xWs2, None, X2, T2, Astar2, None, None, pd2, pd2, it_s2, ok_s2, "Slack-cap equilibrium (fallback).")
    X = xDm + xWm; T = T_of_X(p, X)
    A_D, A_W, A_Wss = thresholds_mixed(p, xDm, xWm, rwm)
    def cdf(z: float) -> float:
        if z <= p.A_low: return 0.0
        if z >= p.A_high: return 1.0
        return (z - p.A_low) / (p.A_high - p.A_low)
    pD = cdf(A_D); pW = cdf(A_W)
    return Eq("mixed", xDm, xWm, rwm, X, T, A_D, A_W, A_Wss, pD, pW, it_m, ok_m, "Mixed regime.")

# Isotonic projection (PAV)
def isotonic_increasing_aligned(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 1:
        return y.copy()
    blocks = [(0, 0, y[0])]
    for k in range(1, n):
        blocks.append((k, k, y[k]))
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2] - 1e-14:
            s1, e1, v1 = blocks[-2]
            s2, e2, v2 = blocks[-1]
            w1 = e1 - s1 + 1
            w2 = e2 - s2 + 1
            v = (w1 * v1 + w2 * v2) / (w1 + w2)
            blocks = blocks[:-2] + [(s1, e2, v)]
    yproj = np.empty(n, dtype=float)
    for s, e, v in blocks:
        yproj[s:e+1] = v
    return yproj

def isotonic_decreasing_aligned(y: np.ndarray) -> np.ndarray:
    return -isotonic_increasing_aligned(-np.asarray(y, dtype=float))

# Piecewise DW sweep
def solve_slack_only(p: Params) -> Eq:
    X0 = (1.0 / p.theta()) ** (1.0 / p.invpow()) if p.theta() > 0.0 else 1.0
    xD0, xW0 = 0.5*X0, 0.5*X0
    xD, xW, ok, it = solve_slack(p, xD0, xW0)
    X = xD + xW; T = T_of_X(p, X)
    Astar = (1.0 - p.gamma)/T
    def cdf(z: float) -> float:
        if z <= p.A_low: return 0.0
        if z >= p.A_high: return 1.0
        return (z - p.A_low) / (p.A_high - p.A_low)
    pd = cdf(Astar)
    return Eq("slack", xD, xW, None, X, T, Astar, None, None, pd, pd, it, ok, "Slack-cap equilibrium.")

def sweep_DW_piecewise(p: Params, DW_grid: List[float], enforce_monotone: bool = True):
    p_slack = Params(**asdict(p)); p_slack.D_W = 1e9
    eq_slack = solve(p_slack)
    DW_star = (1.0 - p.gamma) * eq_slack.x_W

    results: List[Eq] = []
    seed = (eq_slack.x_D, eq_slack.x_W)
    for DW in DW_grid:
        p2 = Params(**asdict(p)); p2.D_W = DW
        if DW >= DW_star - p.cap_deadband:
            e = Eq("slack", eq_slack.x_D, eq_slack.x_W, None,
                   eq_slack.X, eq_slack.T, eq_slack.A_D_star, None, None,
                   eq_slack.pD, eq_slack.pW, eq_slack.iters, True, "Slack (copied)")
        else:
            e = solve(p2, seed)
            seed = (e.x_D, e.x_W)
        results.append(e)

    if not enforce_monotone:
        return results, DW_star

    DW_arr = np.array(DW_grid, dtype=float)
    xW = np.array([e.x_W for e in results], dtype=float)
    xD = np.array([e.x_D for e in results], dtype=float)
    X  = np.array([e.X   for e in results], dtype=float)

    # enforce monotonicity only where wholesale is actually used in the solved results
    W_amt = (1.0 - p.gamma) * np.array([e.x_W for e in results]) - DW_arr
    mask_iso = (W_amt > p.cap_deadband)
    if mask_iso.any():
        xW[mask_iso] = isotonic_increasing_aligned(xW[mask_iso])
        xD[mask_iso] = isotonic_decreasing_aligned(xD[mask_iso])
        X[mask_iso]  = isotonic_increasing_aligned(X[mask_iso])
        adj_results: List[Eq] = []
        for i, DW in enumerate(DW_grid):
            eold = results[i]
            if not mask_iso[i]:
                adj_results.append(eold)
            else:
                p2 = Params(**asdict(p)); p2.D_W = DW
                rw = solve_rw_bisect(p2, xD[i], xW[i])
                Xv = xD[i] + xW[i]; Tv = T_of_X(p2, Xv)
                A_D, A_W, A_Wss = thresholds_mixed(p2, xD[i], xW[i], rw)
                def cdf(z: float) -> float:
                    if z <= p2.A_low: return 0.0
                    if z >= p2.A_high: return 1.0
                    return (z - p2.A_low) / (p2.A_high - p2.A_low)
                pD = cdf(A_D); pW = cdf(A_W)
                adj_results.append(Eq("mixed", xD[i], xW[i], rw, Xv, Tv, A_D, A_W, A_Wss, pD, pW, eold.iters, True, "Mixed (monotone on wholesale-only rows)"))
        results = adj_results

    return results, DW_star

# Calibration helpers
def rebase_L_for_theta_one(p: Params) -> Params:
    target = 1.0
    base = p.pi * (p.alpha ** 2)
    Lnew = (target / base) ** (1.0 / (1.0 - p.alpha))
    q = Params(**asdict(p)); q.L = Lnew
    return q

def acceptance_checks(p: Params, e: Eq) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["theta"] = p.theta()
    out["cap_gap"] = (1.0 - p.gamma) * e.x_W - p.D_W
    out["A_Wss<=A_W*"] = float(e.A_W_star is None or (e.A_W_starstar <= e.A_W_star))
    out["A_W*>A_D* (if mixed)"] = float(e.A_W_star is None or (e.A_W_star > e.A_D_star))
    for k in ["pD","pW"]:
        v = getattr(e, k)
        out[f"{k}_in_[0,1]"] = float((v >= -1e-12) and (v <= 1.0+1e-12))
    return out