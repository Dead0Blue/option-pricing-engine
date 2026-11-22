"""Black-Scholes-Merton closed-form pricing and Greeks for European options.

All functions are vectorised: any argument can be a NumPy array and the
usual broadcasting rules apply.

Conventions
-----------
S      spot price
K      strike
T      time to maturity in years
r      continuously compounded risk-free rate
sigma  volatility (annualised)
q      continuous dividend yield
kind   "call" or "put"

Theta is returned per year (dV/dt with calendar time running forward);
divide by 365 for a daily figure. Vega and rho are per unit change
(divide by 100 for "per vol point" or "per 1% rate").
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def norm_cdf(x):
    return ndtr(x)


def _check_kind(kind: str) -> bool:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind == "call"


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = map(lambda a: np.asarray(a, dtype=float), (S, K, T, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European call or put."""
    is_call = _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * np.asarray(T)), np.exp(-q * np.asarray(T))
    if is_call:
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)
    return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1)


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    """Analytic Greeks: delta, gamma, vega, theta (per year), rho."""
    is_call = _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    df_r, df_q = np.exp(-r * np.asarray(T)), np.exp(-q * np.asarray(T))
    pdf_d1 = norm_pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)
    if is_call:
        delta = df_q * norm_cdf(d1)
        theta = decay - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1)
        rho = K * T * df_r * norm_cdf(d2)
    else:
        delta = df_q * (norm_cdf(d1) - 1.0)
        theta = decay + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1)
        rho = -K * T * df_r * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def parity_gap(call_price, put_price, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}); zero for arbitrage-free prices."""
    return call_price - put_price - (S * np.exp(-q * T) - K * np.exp(-r * T))


def digital_call(S, K, T, r, sigma, q=0.0):
    """Cash-or-nothing digital call paying 1 if S_T > K."""
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-r * np.asarray(T)) * norm_cdf(d2)


def down_and_out_call(S, K, H, T, r, sigma, q=0.0):
    """Continuously monitored down-and-out call with barrier H <= K, no rebate.

    Closed form (Merton 1973, Reiner-Rubinstein 1991): C_do = C - C_di.
    Used as an exact benchmark for the Monte Carlo barrier pricer.
    """
    S, K, H = map(lambda a: np.asarray(a, dtype=float), (S, K, H))
    if np.any(H > K):
        raise ValueError("this closed form requires H <= K")
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    vol_sqrt_t = sigma * np.sqrt(T)
    y = np.log(H**2 / (S * K)) / vol_sqrt_t + lam * vol_sqrt_t
    c_di = (S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm_cdf(y)
            - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm_cdf(y - vol_sqrt_t))
    out = price(S, K, T, r, sigma, q, "call") - c_di
    return np.where(S <= H, 0.0, out)


def geometric_asian_call(S, K, T, r, sigma, n_fixings, q=0.0):
    """Discretely monitored geometric-average Asian call (closed form).

    Fixings at t_i = i T / n, i = 1..n. log G is Gaussian, so the price is a
    Black-Scholes-type formula. Used as a control variate for arithmetic Asians.
    """
    n = n_fixings
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = np.sqrt(var)
    d2 = (mu - np.log(K)) / sd
    d1 = d2 + sd
    return np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm_cdf(d1) - K * norm_cdf(d2))
