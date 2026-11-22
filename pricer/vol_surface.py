"""Implied volatility surface: extract implied vols from an option chain and
fit Gatheral's raw SVI parameterisation slice by slice.

Raw SVI for total implied variance w(k) = sigma_impl(k)^2 * T, with
log-moneyness k = ln(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Five parameters per expiry: level a, slope b >= 0, skew rho in (-1, 1),
shift m, curvature s > 0. SVI is the industry-standard smile shape for
equity indices (Gatheral 2004).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .implied_vol import implied_vol


@dataclass
class SVISlice:
    T: float
    a: float
    b: float
    rho: float
    m: float
    s: float
    rmse_vol: float  # fit error in implied vol units

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2))

    def vol(self, k):
        return np.sqrt(np.maximum(self.total_variance(k), 1e-12) / self.T)

    def butterfly_arbitrage_free(self, k_grid=None) -> bool:
        """Check Gatheral's density condition g(k) >= 0 on a grid."""
        k = np.linspace(-1.5, 1.5, 601) if k_grid is None else k_grid
        w = self.total_variance(k)
        dk = k[1] - k[0]
        w1 = np.gradient(w, dk)
        w2 = np.gradient(w1, dk)
        g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
        return bool(np.all(g > -1e-8))


def svi_raw(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def fit_svi(k, iv, T) -> SVISlice:
    """Least-squares SVI fit of one expiry slice (k = log-moneyness, iv = implied vols)."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w_mkt = iv**2 * T
    atm_w = float(np.interp(0.0, k, w_mkt)) if k.min() < 0 < k.max() else float(w_mkt.mean())
    x0 = [atm_w * 0.5, 0.1, -0.5, 0.0, 0.1]
    lb = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    ub = [2.0, 5.0, 0.999, 1.0, 2.0]

    def resid(p):
        w = svi_raw(k, *p)
        return np.sqrt(np.maximum(w, 1e-12) / T) - iv

    sol = least_squares(resid, x0, bounds=(lb, ub), method="trf")
    rmse = float(np.sqrt(np.mean(sol.fun**2)))
    return SVISlice(T, *map(float, sol.x), rmse_vol=rmse)


def implied_vols_from_chain(S, r, q, strikes, maturities, prices, kinds):
    """Vectorised implied vols for a chain given as parallel arrays."""
    strikes, maturities, prices = (np.asarray(a, float) for a in (strikes, maturities, prices))
    kinds = np.asarray(kinds)
    iv = np.full(prices.shape, np.nan)
    for kind in ("call", "put"):
        mask = kinds == kind
        if mask.any():
            iv[mask] = implied_vol(prices[mask], S, strikes[mask], maturities[mask], r, q, kind)
    return iv


def fit_surface(S, r, q, strikes, maturities, prices, kinds):
    """Implied vols for every quote, then one SVI fit per maturity."""
    iv = implied_vols_from_chain(S, r, q, strikes, maturities, prices, kinds)
    maturities = np.asarray(maturities, float)
    strikes = np.asarray(strikes, float)
    slices = []
    for T in np.unique(maturities):
        mask = (maturities == T) & np.isfinite(iv)
        if mask.sum() < 6:
            continue
        fwd = S * np.exp((r - q) * T)
        k = np.log(strikes[mask] / fwd)
        order = np.argsort(k)
        slices.append(fit_svi(k[order], iv[mask][order], T))
    return iv, slices
