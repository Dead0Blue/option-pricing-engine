"""Monte Carlo pricing under geometric Brownian motion (risk-neutral measure).

Features
- European options with plain, antithetic and control-variate estimators,
  each returning a price, a standard error and a 95% confidence interval.
- Pathwise delta estimator.
- Arithmetic Asian calls, with the closed-form geometric Asian as a
  control variate (Kemna-Vorst).
- Discretely simulated down-and-out calls with a Brownian-bridge crossing
  correction, which removes the bias of checking the barrier only on the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs


@dataclass
class MCResult:
    price: float
    stderr: float
    n_paths: int

    @property
    def ci95(self):
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)

    def __repr__(self):
        lo, hi = self.ci95
        return f"MCResult(price={self.price:.5f}, stderr={self.stderr:.5f}, ci95=[{lo:.5f}, {hi:.5f}])"


def _terminal(S, T, r, sigma, q, z):
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _result(samples) -> MCResult:
    n = samples.size
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n)), n)


def european(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
             method="plain", seed=None) -> MCResult:
    """European option by Monte Carlo.

    method = "plain" | "antithetic" | "control"
    The control variate is the discounted terminal stock price, whose
    risk-neutral expectation S e^{-qT} is known exactly.
    """
    rng = np.random.default_rng(seed)
    sign = 1.0 if kind == "call" else -1.0
    disc = np.exp(-r * T)

    if method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        pay_plus = np.maximum(sign * (_terminal(S, T, r, sigma, q, z) - K), 0.0)
        pay_minus = np.maximum(sign * (_terminal(S, T, r, sigma, q, -z) - K), 0.0)
        return _result(disc * 0.5 * (pay_plus + pay_minus))

    z = rng.standard_normal(n_paths)
    s_t = _terminal(S, T, r, sigma, q, z)
    pay = disc * np.maximum(sign * (s_t - K), 0.0)
    if method == "plain":
        return _result(pay)
    if method == "control":
        control = disc * s_t
        control_mean = S * np.exp(-q * T)
        cov = np.cov(pay, control)
        beta = cov[0, 1] / cov[1, 1]
        return _result(pay - beta * (control - control_mean))
    raise ValueError("method must be 'plain', 'antithetic' or 'control'")


def pathwise_delta(S, K, T, r, sigma, q=0.0, kind="call", n_paths=200_000, seed=None) -> MCResult:
    """Pathwise (IPA) delta: d payoff / d S0 = 1{ITM} * sign * S_T / S0, discounted."""
    rng = np.random.default_rng(seed)
    s_t = _terminal(S, T, r, sigma, q, rng.standard_normal(n_paths))
    sign = 1.0 if kind == "call" else -1.0
    itm = sign * (s_t - K) > 0
    return _result(np.exp(-r * T) * itm * sign * s_t / S)


def simulate_paths(S, T, r, sigma, q=0.0, n_steps=252, n_paths=10_000, seed=None,
                   drift=None):
    """GBM paths on an even time grid. Returns array of shape (n_paths, n_steps + 1).

    drift: real-world drift mu; defaults to the risk-neutral drift r - q.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu = (r - q) if drift is None else drift
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S * np.exp(log_paths)


def asian_arithmetic_call(S, K, T, r, sigma, q=0.0, n_fixings=12, n_paths=100_000,
                          control_variate=True, seed=None) -> MCResult:
    """Arithmetic-average Asian call with fixings at T/n, 2T/n, ..., T."""
    paths = simulate_paths(S, T, r, sigma, q, n_fixings, n_paths, seed)[:, 1:]
    disc = np.exp(-r * T)
    arith = disc * np.maximum(paths.mean(axis=1) - K, 0.0)
    if not control_variate:
        return _result(arith)
    geo = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - K, 0.0)
    geo_exact = bs.geometric_asian_call(S, K, T, r, sigma, n_fixings, q)
    cov = np.cov(arith, geo)
    beta = cov[0, 1] / cov[1, 1]
    return _result(arith - beta * (geo - geo_exact))


def down_and_out_call(S, K, H, T, r, sigma, q=0.0, n_steps=50, n_paths=100_000,
                      bridge=True, seed=None) -> MCResult:
    """Down-and-out call, continuously monitored barrier H < S.

    Without the bridge correction, a path that dips below H between two grid
    dates is missed, so the price is biased upwards. With the correction,
    each step survives with probability 1 - exp(-2 a b / (sigma^2 dt)),
    where a and b are the log-distances to the barrier at the two ends.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_h = np.log(H)
    x = np.full(n_paths, np.log(S))
    alive = np.ones(n_paths)  # survival weight (probability of not having crossed)
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    for _ in range(n_steps):
        x_new = x + drift + vol * rng.standard_normal(n_paths)
        above = x_new > log_h
        if bridge:
            a, b = x - log_h, x_new - log_h
            p_cross = np.where(above, np.exp(-2.0 * a * b / (sigma**2 * dt)), 1.0)
            alive *= 1.0 - p_cross
        else:
            alive *= above
        x = x_new
    pay = np.exp(-r * T) * alive * np.maximum(np.exp(x) - K, 0.0)
    return _result(pay)
