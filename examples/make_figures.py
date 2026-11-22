"""Reproduce every figure and the results table in the README.

    python examples/make_figures.py

Writes PNGs to docs/figures/ and a markdown table to docs/results.md.
"""
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricer import binomial, hedging, monte_carlo as mc, pde
from pricer import black_scholes as bs
from pricer.vol_surface import fit_surface

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


rows = []  # results table

# 1. Convergence of numerical methods to Black-Scholes -------------------------
exact = float(bs.price(S, K, T, R, SIGMA))
steps = np.arange(10, 401, 5)
tree_err = [binomial.crr_price(S, K, T, R, SIGMA, steps=n).price - exact for n in steps]
paths = np.unique(np.logspace(3, 6, 13).astype(int))
se = {m: [mc.european(S, K, T, R, SIGMA, n_paths=n, method=m, seed=1).stderr for n in paths]
      for m in ("plain", "antithetic", "control")}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(steps, tree_err, color=C1)
ax[0].axhline(0, color=MUTED, lw=1)
ax[0].set(title="Binomial tree error vs number of steps", xlabel="steps", ylabel="CRR price - Black-Scholes")
for m, col in zip(se, (C1, C2, C3)):
    ax[1].loglog(paths, se[m], color=col, marker="o", ms=4, label=m)
ax[1].loglog(paths, 15 / np.sqrt(paths), color=MUTED, ls="--", lw=1, label=r"slope $-1/2$")
ax[1].set(title="Monte Carlo standard error vs paths", xlabel="paths", ylabel="standard error")
ax[1].legend()
save(fig, "convergence.png")

t0 = time.perf_counter(); tree = binomial.crr_price(S, K, T, R, SIGMA, steps=2000); t_tree = time.perf_counter() - t0
t0 = time.perf_counter(); fd = pde.price(S, K, T, R, SIGMA); t_pde = time.perf_counter() - t0
t0 = time.perf_counter(); mcp = mc.european(S, K, T, R, SIGMA, n_paths=1_000_000, method="control", seed=2026); t_mc = time.perf_counter() - t0
rows += [("European call, Black-Scholes closed form", exact, None, None),
         ("European call, CRR tree (2,000 steps)", tree.price, None, t_tree),
         ("European call, Crank-Nicolson PDE (400x400)", fd, None, t_pde),
         ("European call, Monte Carlo + control variate (1M paths)", mcp.price, mcp.stderr, t_mc)]

# 2. Greeks across spot -----------------------------------------------------------
spots = np.linspace(60, 140, 400)
fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
for tau, col in zip((0.1, 0.5, 1.0), (C1, C2, C3)):
    g = bs.greeks(spots, K, tau, R, SIGMA)
    label = f"T = {tau:g}y"
    ax[0, 0].plot(spots, g["delta"], color=col, label=label)
    ax[0, 1].plot(spots, g["gamma"], color=col, label=label)
    ax[1, 0].plot(spots, g["vega"] / 100, color=col, label=label)
    ax[1, 1].plot(spots, g["theta"] / 365, color=col, label=label)
for a, title in zip(ax.flat, ("Delta", "Gamma", "Vega (per vol point)", "Theta (per day)")):
    a.set(title=f"Call {title}", xlabel="spot")
    a.axvline(K, color=MUTED, lw=1, ls=":")
ax[0, 0].legend()
save(fig, "greeks.png")

# 3. American put: early-exercise premium and boundary --------------------------
am = binomial.crr_price(S, K, T, R, SIGMA, kind="put", american=True, steps=2000, return_boundary=True)
eu_put = float(bs.price(S, K, T, R, SIGMA, kind="put"))
am_pde = pde.price(S, K, T, R, SIGMA, kind="put", american=True, n_space=800, n_time=800)
rows += [("European put, Black-Scholes", eu_put, None, None),
         ("American put, CRR tree (2,000 steps)", am.price, None, None),
         ("American put, PDE with early exercise", am_pde, None, None)]
fig, ax = plt.subplots(figsize=(6, 3.8))
bt, bsp = am.exercise_boundary
ax.plot(bt, bsp, color=C1)
ax.fill_between(bt, 0, bsp, color=C1, alpha=0.12)
ax.text(0.45, 70, "exercise now", color=INK)
ax.text(0.45, 92, "keep holding", color=INK)
ax.set(title="American put: early-exercise boundary (K = 100)", xlabel="time (years)",
       ylabel="spot", ylim=(60, 100))
save(fig, "american_boundary.png")

# 4. Exotics: variance reduction and barrier bias -------------------------------
asian_cv = mc.asian_arithmetic_call(S, K, T, R, SIGMA, n_paths=200_000, seed=1)
asian_plain = mc.asian_arithmetic_call(S, K, T, R, SIGMA, n_paths=200_000, control_variate=False, seed=1)
H = 90.0
doc_exact = float(bs.down_and_out_call(S, K, H, T, R, SIGMA))
doc_bridge = mc.down_and_out_call(S, K, H, T, R, SIGMA, n_steps=50, n_paths=400_000, seed=1)
doc_naive = mc.down_and_out_call(S, K, H, T, R, SIGMA, n_steps=50, n_paths=400_000, bridge=False, seed=1)
rows += [("Asian call (12 fixings), plain MC", asian_plain.price, asian_plain.stderr, None),
         ("Asian call, geometric control variate", asian_cv.price, asian_cv.stderr, None),
         ("Down-and-out call (H = 90), closed form", doc_exact, None, None),
         ("Down-and-out call, MC grid monitoring (biased)", doc_naive.price, doc_naive.stderr, None),
         ("Down-and-out call, MC + Brownian bridge", doc_bridge.price, doc_bridge.stderr, None)]

n_grid = [5, 10, 25, 50, 100, 250]
naive_b = [mc.down_and_out_call(S, K, H, T, R, SIGMA, n_steps=n, n_paths=200_000, bridge=False, seed=2).price for n in n_grid]
bridge_b = [mc.down_and_out_call(S, K, H, T, R, SIGMA, n_steps=n, n_paths=200_000, seed=2).price for n in n_grid]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(n_grid, naive_b, color=C2, marker="o", ms=5, label="grid monitoring")
ax.semilogx(n_grid, bridge_b, color=C1, marker="o", ms=5, label="Brownian bridge")
ax.axhline(doc_exact, color=INK, lw=1, ls="--", label="closed form")
ax.set(title="Down-and-out call: barrier bias", xlabel="time steps", ylabel="price")
ax.legend()
save(fig, "barrier_bias.png")

# 5. Delta hedging -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for real, col, lab in ((0.15, C1, r"realised 15%"), (0.20, C3, r"realised 20%"), (0.25, C2, r"realised 25%")):
    h = hedging.short_call_delta_hedge(S, K, T, R, 0.20, real, mu=0.08, n_rebalance=252, n_paths=20_000, seed=4)
    ax[0].hist(h.pnl, bins=80, color=col, alpha=0.55, label=f"{lab}: mean {h.mean:+.2f}")
ax[0].axvline(0, color=INK, lw=1)
ax[0].set(title="Short call sold at 20% implied vol, daily hedge", xlabel="P&L at expiry", ylabel="paths")
ax[0].legend()
freqs = np.array([4, 12, 26, 52, 126, 252, 504])
stds = [hedging.short_call_delta_hedge(S, K, T, R, SIGMA, SIGMA, mu=0.08, n_rebalance=n, n_paths=20_000, seed=5).std for n in freqs]
ax[1].loglog(freqs, stds, color=C1, marker="o", ms=5, label="simulated")
ax[1].loglog(freqs, stds[0] * np.sqrt(freqs[0] / freqs), color=MUTED, ls="--", lw=1, label=r"$\propto 1/\sqrt{N}$")
ax[1].set(title="Hedging error vs rebalancing frequency", xlabel="rebalances per year", ylabel="std of P&L")
ax[1].legend()
save(fig, "hedging.png")

h_mis = hedging.short_call_delta_hedge(S, K, T, R, 0.25, 0.18, n_rebalance=252, n_paths=20_000, seed=6)
theory = hedging.gamma_pnl_prediction(S, K, T, R, 0.25, 0.18, n_paths=20_000, seed=7)
rows += [("Hedged short call, implied 25% / realised 18%: simulated mean P&L", h_mis.mean, h_mis.std / np.sqrt(h_mis.pnl.size), None),
         ("Same, gamma P&L theory 1/2 Gamma S^2 (sigma_i^2 - sigma_r^2)", theory, None, None)]

# 6. Volatility surface (synthetic chain) ----------------------------------------
with open("data/chain_synthetic.csv") as f:
    chain = list(csv.DictReader(f))
spot, r, q = float(chain[0]["spot"]), float(chain[0]["r"]), float(chain[0]["q"])
K_arr = np.array([float(c["strike"]) for c in chain])
T_arr = np.array([float(c["T"]) for c in chain])
P_arr = np.array([float(c["price"]) for c in chain])
kinds = np.array([c["kind"] for c in chain])
iv, slices = fit_surface(spot, r, q, K_arr, T_arr, P_arr, kinds)
fig, ax = plt.subplots(figsize=(7, 4.2))
cols = [C1, C2, C3, C4, "#6250d6"]
for sl, col in zip(slices, cols):
    m = T_arr == sl.T
    k = np.log(K_arr[m] / (spot * np.exp((r - q) * sl.T)))
    kk = np.linspace(k.min(), k.max(), 200)
    ax.scatter(k, iv[m] * 100, color=col, s=14, zorder=3)
    ax.plot(kk, sl.vol(kk) * 100, color=col, label=f"T = {sl.T:.2f}y  (fit RMSE {sl.rmse_vol*1e4:.0f}bp)")
ax.set(title="Implied volatility smile: quotes (dots) and SVI fits (lines)", xlabel="log-moneyness ln(K/F)",
       ylabel="implied vol (%)")
ax.legend(fontsize=8)
save(fig, "vol_surface.png")
arb_free = all(sl.butterfly_arbitrage_free() for sl in slices)

# Results table -------------------------------------------------------------------
lines = ["| Instrument and method | Price | Std. error | Time |", "|---|---:|---:|---:|"]
for name, p, err, t in rows:
    lines.append(f"| {name} | {p:.4f} | {'' if err is None else f'{err:.4f}'} | {'' if t is None else f'{t*1000:.0f} ms'} |")
lines.append("")
lines.append(f"Asian control variate: standard error divided by {asian_plain.stderr / asian_cv.stderr:.0f} for the same number of paths.")
zs = [(lambda res: (res.price - exact) / res.stderr)(mc.european(S, K, T, R, SIGMA, n_paths=200_000, method="control", seed=sd)) for sd in range(100)]
lines.append(f"Monte Carlo calibration: over 100 seeds, (MC - exact) / stderr has mean {np.mean(zs):+.2f} and standard deviation {np.std(zs):.2f} (unbiased estimator: 0 and 1).")
lines.append(f"SVI: {len(slices)} maturities fitted, all slices butterfly-arbitrage-free: {arb_free}.")
Path("docs/results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
