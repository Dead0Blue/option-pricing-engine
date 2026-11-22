"""pricer: an option pricing engine.

Modules
- black_scholes : closed-form prices, Greeks, digital, barrier, geometric Asian
- implied_vol   : safeguarded Newton implied volatility solver
- binomial      : Cox-Ross-Rubinstein trees, American exercise, exercise boundary
- monte_carlo   : European (plain/antithetic/control variate), pathwise delta,
                  arithmetic Asian with geometric control variate,
                  barrier with Brownian-bridge correction
- pde           : Crank-Nicolson finite differences with Rannacher smoothing
- hedging       : discrete delta-hedging P&L simulator
- vol_surface   : implied vol surface and SVI calibration
"""
from . import binomial, black_scholes, hedging, implied_vol, monte_carlo, pde, vol_surface

__all__ = ["black_scholes", "implied_vol", "binomial", "monte_carlo", "pde", "hedging", "vol_surface"]
__version__ = "1.0.0"
