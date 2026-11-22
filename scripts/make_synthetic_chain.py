"""Generate a SYNTHETIC equity-index option chain, so the vol surface example
runs offline. Prices come from SVI smiles with a realistic downside skew,
plus bid-ask-sized noise. Replace with real data via scripts/fetch_chain.py.
"""
import csv

import numpy as np

from pricer import black_scholes as bs
from pricer.vol_surface import svi_raw

SPOT, R, Q = 100.0, 0.04, 0.013
EXPIRIES = [1 / 12, 0.25, 0.5, 1.0, 2.0]
rng = np.random.default_rng(2026)

rows = []
for T in EXPIRIES:
    atm_var = 0.18**2 * T
    params = dict(a=0.6 * atm_var, b=0.08 * np.sqrt(T) + 0.02, rho=-0.7, m=0.01, s=0.12 * np.sqrt(T) + 0.03)
    fwd = SPOT * np.exp((R - Q) * T)
    for K in np.arange(60.0, 141.0, 2.5):
        k = np.log(K / fwd)
        if abs(k) > 2.2 * 0.2 * np.sqrt(T) + 0.05:
            continue
        vol = np.sqrt(svi_raw(k, **params) / T)
        kind = "call" if K >= SPOT else "put"
        p = float(bs.price(SPOT, K, T, R, vol, Q, kind))
        p = max(p * (1 + rng.normal(0, 0.004)) + rng.normal(0, 0.005), 0.01)
        rows.append(dict(strike=K, T=round(T, 6), price=round(p, 4), kind=kind, spot=SPOT, r=R, q=Q))

with open("data/chain_synthetic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} synthetic quotes to data/chain_synthetic.csv")
