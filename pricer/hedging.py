"""Discrete delta-hedging simulator.

A trader sells a European call at the Black-Scholes price computed with an
implied volatility sigma_impl, then delta-hedges with shares, rebalancing
n_rebalance times until expiry. The stock actually follows GBM with a
real-world drift mu and a realised volatility sigma_real.

Theory checked by the simulations:
- If sigma_real == sigma_impl, the hedged P&L has mean ~0 and a standard
  deviation that shrinks like 1/sqrt(n_rebalance) (discrete hedging error).
- If sigma_real != sigma_impl, the seller's P&L is approximately
      integral of 1/2 * Gamma * S^2 * (sigma_impl^2 - sigma_real^2) dt,
  positive when the market turns out calmer than priced.
- The drift mu does not matter on average: hedging removes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from .monte_carlo import simulate_paths


@dataclass
class HedgeResult:
    pnl: np.ndarray          # final P&L of the short-option hedged book, per path
    premium: float           # price received for the option

    @property
    def mean(self):
        return float(self.pnl.mean())

    @property
    def std(self):
        return float(self.pnl.std(ddof=1))


def short_call_delta_hedge(S, K, T, r, sigma_impl, sigma_real, mu=None, q=0.0,
                           n_rebalance=252, n_paths=20_000, hedge_vol=None,
                           cost_per_share=0.0, seed=None) -> HedgeResult:
    """Sell one call, delta-hedge discretely, return the P&L distribution at expiry.

    hedge_vol: volatility used to compute the hedge ratio (defaults to sigma_impl).
    cost_per_share: proportional transaction cost, as a fraction of the stock price.
    """
    hedge_vol = sigma_impl if hedge_vol is None else hedge_vol
    mu = r if mu is None else mu
    dt = T / n_rebalance
    paths = simulate_paths(S, T, r, sigma_real, q, n_rebalance, n_paths, seed, drift=mu)

    premium = float(bs.price(S, K, T, r, sigma_impl, q, "call"))
    delta = bs.greeks(S, K, T, r, hedge_vol, q, "call")["delta"]
    shares = np.full(n_paths, delta)
    cash = premium - shares * S - cost_per_share * np.abs(shares) * S

    for i in range(1, n_rebalance):
        s_i = paths[:, i]
        cash = cash * np.exp(r * dt) + shares * s_i * (np.exp(q * dt) - 1.0)  # interest + dividends
        tau = T - i * dt
        new_shares = bs.greeks(s_i, K, tau, r, hedge_vol, q, "call")["delta"]
        trade = new_shares - shares
        cash -= trade * s_i + cost_per_share * np.abs(trade) * s_i
        shares = new_shares

    s_T = paths[:, -1]
    cash = cash * np.exp(r * dt) + shares * s_T * (np.exp(q * dt) - 1.0)
    pnl = cash + shares * s_T - np.maximum(s_T - K, 0.0)
    return HedgeResult(pnl=pnl, premium=premium)


def gamma_pnl_prediction(S, K, T, r, sigma_impl, sigma_real, q=0.0, n_steps=252,
                         n_paths=20_000, seed=None):
    """Expected seller P&L predicted by 1/2 Gamma S^2 (sigma_impl^2 - sigma_real^2) dt.

    Integrated along simulated paths; returns the average across paths
    (future-valued to expiry). Used to check the simulator against theory.
    """
    dt = T / n_steps
    paths = simulate_paths(S, T, r, sigma_real, q, n_steps, n_paths, seed)
    total = np.zeros(n_paths)
    for i in range(n_steps):
        tau = T - i * dt
        g = bs.greeks(paths[:, i], K, tau, r, sigma_impl, q, "call")["gamma"]
        total += 0.5 * g * paths[:, i] ** 2 * (sigma_impl**2 - sigma_real**2) * dt * np.exp(r * tau)
    return float(total.mean())
