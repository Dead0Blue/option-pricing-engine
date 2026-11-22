"""Finite-difference solver for the Black-Scholes PDE.

We solve in log-spot x = ln S and time-to-maturity tau = T - t:

    V_tau = 1/2 sigma^2 V_xx + (r - q - 1/2 sigma^2) V_x - r V

with a theta-scheme on a uniform grid. Crank-Nicolson (theta = 1/2) is
second order in time and space; the first few steps use fully implicit
Euler (Rannacher smoothing) to damp the oscillations caused by the kink of
the payoff at the strike. Each step solves a tridiagonal system in O(N).
American exercise is handled by projecting on the payoff after each step.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def price(S, K, T, r, sigma, q=0.0, kind="call", american=False, n_space=400,
          n_time=400, width=5.0, rannacher_steps=4):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    sign = 1.0 if kind == "call" else -1.0

    x0 = np.log(S)
    half = width * sigma * np.sqrt(T)
    x = np.linspace(x0 - half, x0 + half, n_space + 1)
    h = x[1] - x[0]
    dt = T / n_time
    spot = np.exp(x)
    payoff = np.maximum(sign * (spot - K), 0.0)
    v = payoff.copy()

    # Spatial operator L V = a V_{j-1} + b V_j + c V_{j+1}
    nu = r - q - 0.5 * sigma**2
    a = 0.5 * sigma**2 / h**2 - 0.5 * nu / h
    b = -sigma**2 / h**2 - r
    c = 0.5 * sigma**2 / h**2 + 0.5 * nu / h

    def boundaries(tau):
        if kind == "call":
            lo = 0.0
            hi = spot[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
        else:
            lo = K * np.exp(-r * tau) - spot[0] * np.exp(-q * tau)
            hi = 0.0
            if american:
                lo = K - spot[0]
        return lo, hi

    n_in = n_space - 1
    for step in range(1, n_time + 1):
        theta = 1.0 if step <= rannacher_steps else 0.5
        tau = step * dt
        # (I - theta dt L) V^{n+1} = (I + (1-theta) dt L) V^n
        rhs = v[1:-1] + (1 - theta) * dt * (a * v[:-2] + b * v[1:-1] + c * v[2:])
        lo, hi = boundaries(tau)
        rhs[0] += theta * dt * a * lo
        rhs[-1] += theta * dt * c * hi
        ab = np.zeros((3, n_in))
        ab[0, 1:] = -theta * dt * c
        ab[1, :] = 1 - theta * dt * b
        ab[2, :-1] = -theta * dt * a
        inner = solve_banded((1, 1), ab, rhs)
        v = np.concatenate([[lo], inner, [hi]])
        if american:
            v = np.maximum(v, payoff)

    return float(np.interp(x0, x, v))
