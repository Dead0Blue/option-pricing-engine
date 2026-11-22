"""Tests: every pricing method is checked against an independent one."""
import numpy as np
import pytest

from pricer import binomial, hedging, monte_carlo as mc, pde
from pricer import black_scholes as bs
from pricer.implied_vol import implied_vol
from pricer.vol_surface import fit_svi, svi_raw

S, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


# ---------- Black-Scholes ----------
def test_reference_values():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs.price(42, 40, 0.5, 0.10, 0.20, kind="call") == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, kind="put") == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    strikes = np.linspace(60, 140, 17)
    c = bs.price(S, strikes, T, R, SIGMA, q, "call")
    p = bs.price(S, strikes, T, R, SIGMA, q, "put")
    assert np.allclose(bs.parity_gap(c, p, S, strikes, T, R, q), 0.0, atol=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_greeks_match_finite_differences(kind, q):
    g = bs.greeks(S, 95.0, T, R, SIGMA, q, kind)
    f = lambda **kw: bs.price(**{**dict(S=S, K=95.0, T=T, r=R, sigma=SIGMA, q=q, kind=kind), **kw})
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=SIGMA + h) - f(sigma=SIGMA - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-6)
    # theta = dV/dt = -dV/dT
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)


def test_bs_pde_holds():
    g = bs.greeks(S, K, T, R, SIGMA)
    v = bs.price(S, K, T, R, SIGMA)
    lhs = g["theta"] + 0.5 * SIGMA**2 * S**2 * g["gamma"] + R * S * g["delta"] - R * v
    assert lhs == pytest.approx(0.0, abs=1e-10)


def test_price_bounds_and_monotonicity():
    strikes = np.linspace(50, 150, 101)
    c = bs.price(S, strikes, T, R, SIGMA)
    assert np.all(c <= S) and np.all(c >= np.maximum(S - strikes * np.exp(-R * T), 0) - 1e-12)
    assert np.all(np.diff(c) < 0)                 # decreasing in strike
    assert np.all(np.diff(c, 2) > -1e-12)         # convex in strike (no butterfly arbitrage)


# ---------- Implied volatility ----------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    strikes = np.array([40, 70, 90, 100, 110, 140, 250.0])
    vols = np.array([0.6, 0.35, 0.25, 0.2, 0.18, 0.22, 0.9])
    prices = bs.price(S, strikes, T, R, vols, kind=kind)
    assert np.allclose(implied_vol(prices, S, strikes, T, R, kind=kind), vols, atol=1e-7)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(150.0, S, K, T, R, kind="call"))   # above the stock price
    assert np.isnan(implied_vol(0.0, S, K, T, R, kind="call"))     # below intrinsic value


# ---------- Binomial ----------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_crr_converges_to_bs(kind):
    tree = binomial.crr_price(S, K, T, R, SIGMA, kind=kind, steps=2000)
    assert tree.price == pytest.approx(bs.price(S, K, T, R, SIGMA, kind=kind), abs=2e-3)
    assert tree.delta == pytest.approx(bs.greeks(S, K, T, R, SIGMA, kind=kind)["delta"], abs=2e-3)


def test_american_call_equals_european_without_dividends():
    am = binomial.crr_price(S, K, T, R, SIGMA, kind="call", american=True, steps=1000).price
    eu = binomial.crr_price(S, K, T, R, SIGMA, kind="call", american=False, steps=1000).price
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_put_early_exercise_premium():
    am = binomial.crr_price(S, K, T, R, SIGMA, kind="put", american=True, steps=1000)
    assert am.price > bs.price(S, K, T, R, SIGMA, kind="put") + 0.3
    assert am.price >= K - S


# ---------- PDE ----------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_crank_nicolson_matches_bs(kind):
    assert pde.price(S, K, T, R, SIGMA, kind=kind) == pytest.approx(
        bs.price(S, K, T, R, SIGMA, kind=kind), abs=2e-3)


def test_pde_and_tree_agree_on_american_put():
    tree = binomial.crr_price(S, K, T, R, SIGMA, kind="put", american=True, steps=4000).price
    assert pde.price(S, K, T, R, SIGMA, kind="put", american=True, n_space=800, n_time=800) == pytest.approx(tree, abs=5e-3)


# ---------- Monte Carlo ----------
@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
def test_mc_european_within_confidence_interval(method):
    res = mc.european(S, K, T, R, SIGMA, n_paths=400_000, method=method, seed=7)
    assert abs(res.price - bs.price(S, K, T, R, SIGMA)) < 4 * res.stderr


def test_variance_reduction_reduces_error():
    plain = mc.european(S, K, T, R, SIGMA, n_paths=200_000, method="plain", seed=3)
    control = mc.european(S, K, T, R, SIGMA, n_paths=200_000, method="control", seed=3)
    assert control.stderr < 0.5 * plain.stderr


def test_pathwise_delta():
    res = mc.pathwise_delta(S, K, T, R, SIGMA, n_paths=400_000, seed=11)
    assert abs(res.price - bs.greeks(S, K, T, R, SIGMA)["delta"]) < 4 * res.stderr


def test_asian_control_variate():
    cv = mc.asian_arithmetic_call(S, K, T, R, SIGMA, n_paths=100_000, seed=5)
    plain = mc.asian_arithmetic_call(S, K, T, R, SIGMA, n_paths=100_000, control_variate=False, seed=5)
    assert cv.stderr < plain.stderr / 10
    assert abs(cv.price - plain.price) < 4 * plain.stderr
    # arithmetic average >= geometric average
    assert cv.price > bs.geometric_asian_call(S, K, T, R, SIGMA, 12)


def test_barrier_bridge_correction_removes_bias():
    exact = bs.down_and_out_call(S, K, 90.0, T, R, SIGMA)
    bridge = mc.down_and_out_call(S, K, 90.0, T, R, SIGMA, n_steps=50, n_paths=300_000, seed=9)
    naive = mc.down_and_out_call(S, K, 90.0, T, R, SIGMA, n_steps=50, n_paths=300_000, bridge=False, seed=9)
    assert abs(bridge.price - exact) < 4 * bridge.stderr
    assert naive.price - exact > 5 * naive.stderr      # grid monitoring overprices


# ---------- Hedging ----------
def test_hedging_error_shrinks_with_rebalancing():
    coarse = hedging.short_call_delta_hedge(S, K, T, R, SIGMA, SIGMA, mu=0.1, n_rebalance=25, n_paths=20_000, seed=1)
    fine = hedging.short_call_delta_hedge(S, K, T, R, SIGMA, SIGMA, mu=0.1, n_rebalance=400, n_paths=20_000, seed=1)
    assert abs(fine.mean) < 0.05
    assert fine.std < 0.4 * coarse.std          # ~ 1/sqrt(16) = 0.25 in theory


def test_vol_misspecification_pnl_matches_gamma_theory():
    res = hedging.short_call_delta_hedge(S, K, T, R, 0.25, 0.18, n_rebalance=252, n_paths=20_000, seed=2)
    theory = hedging.gamma_pnl_prediction(S, K, T, R, 0.25, 0.18, n_paths=20_000, seed=3)
    assert res.mean > 0
    assert res.mean == pytest.approx(theory, rel=0.05)


# ---------- SVI ----------
def test_svi_recovers_parameters():
    true = dict(a=0.02, b=0.12, rho=-0.6, m=0.02, s=0.15)
    k = np.linspace(-0.5, 0.4, 25)
    iv = np.sqrt(svi_raw(k, **true) / 0.5)
    fit = fit_svi(k, iv, 0.5)
    assert fit.rmse_vol < 1e-6
    assert fit.rho == pytest.approx(true["rho"], abs=1e-3)
    assert fit.butterfly_arbitrage_free()
