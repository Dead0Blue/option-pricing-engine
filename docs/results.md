| Instrument and method | Price | Std. error | Time |
|---|---:|---:|---:|
| European call, Black-Scholes closed form | 10.4506 |  |  |
| European call, CRR tree (2,000 steps) | 10.4496 |  | 45 ms |
| European call, Crank-Nicolson PDE (400x400) | 10.4500 |  | 18 ms |
| European call, Monte Carlo + control variate (1M paths) | 10.4493 | 0.0056 | 37 ms |
| European put, Black-Scholes | 5.5735 |  |  |
| American put, CRR tree (2,000 steps) | 6.0900 |  |  |
| American put, PDE with early exercise | 6.0894 |  |  |
| Asian call (12 fixings), plain MC | 6.1644 | 0.0190 |  |
| Asian call, geometric control variate | 6.1557 | 0.0005 |  |
| Down-and-out call (H = 90), closed form | 8.6655 |  |  |
| Down-and-out call, MC grid monitoring (biased) | 9.2077 | 0.0233 |  |
| Down-and-out call, MC + Brownian bridge | 8.6936 | 0.0229 |  |
| Hedged short call, implied 25% / realised 18%: simulated mean P&L | 2.7666 | 0.0068 |  |
| Same, gamma P&L theory 1/2 Gamma S^2 (sigma_i^2 - sigma_r^2) | 2.7699 |  |  |

Asian control variate: standard error divided by 36 for the same number of paths.
Monte Carlo calibration: over 100 seeds, (MC - exact) / stderr has mean -0.00 and standard deviation 0.98 (unbiased estimator: 0 and 1).
SVI: 5 maturities fitted, all slices butterfly-arbitrage-free: True.
