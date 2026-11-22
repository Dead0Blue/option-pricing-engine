"""Implied volatility: invert Black-Scholes for sigma.

Safeguarded Newton-Raphson, vectorised over arrays of options. Each Newton
step that would leave the current bracket [lo, hi] is replaced by a
bisection step, so the solver converges even far out of the money, where
vega is tiny and plain Newton diverges.
"""
from __future__ import annotations

import numpy as np

from . import black_scholes as bs


def no_arbitrage_bounds(S, K, T, r, q=0.0, kind="call"):
    """Lower and upper bounds that any arbitrage-free option price must respect."""
    fwd_s, fwd_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return np.maximum(fwd_s - fwd_k, 0.0), fwd_s
    return np.maximum(fwd_k - fwd_s, 0.0), fwd_k


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=100,
                lo=1e-6, hi=5.0):
    """Implied volatility of European option prices.

    Returns NaN where the price violates no-arbitrage bounds (no solution).
    """
    price, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                                 for a in (price, S, K, T, r, q)))
    out = np.full(price.shape, np.nan)
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, kind)
    ok = (price > lower) & (price < upper)
    if not ok.any():
        return out if out.ndim else float(out)

    p, s, k, t, rr, qq = (a[ok] for a in (price, S, K, T, r, q))
    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)
    # Initial guess: Brenner-Subrahmanyam ATM approximation, clipped to the bracket.
    sigma = np.clip(np.sqrt(2 * np.pi / t) * p / s, 0.05, 2.0)

    for _ in range(max_iter):
        diff = bs.price(s, k, t, rr, sigma, qq, kind) - p
        if np.all(np.abs(diff) < tol):
            break
        # Price is increasing in sigma: update the bracket.
        a = np.where(diff < 0, sigma, a)
        b = np.where(diff > 0, sigma, b)
        vega = bs.greeks(s, k, t, rr, sigma, qq, kind)["vega"]
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = sigma - diff / vega
        use_newton = np.isfinite(newton) & (newton > a) & (newton < b)
        sigma = np.where(use_newton, newton, 0.5 * (a + b))

    out[ok] = sigma
    return out if out.ndim else float(out)
