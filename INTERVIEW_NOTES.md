# Interview notes: how to present and defend this project

Read this before putting the project on your CV. An interviewer can open any file and ask "why?". You should be able to answer every question below without looking.

## Your 60-second pitch
"I built an option pricing library in Python with a C++ core. It prices European, American and exotic options with four independent methods: the Black-Scholes formula, binomial trees, finite differences and Monte Carlo. I validate each method against the others with automated tests. On top of that I added variance reduction, a Brownian-bridge correction for barrier options, an implied volatility solver with SVI calibration of the smile, and a delta-hedging simulator. It shows that a hedged option seller earns ½ Γ S² (σ_implied² − σ_realised²), matching theory to within 0.2%. The C++ engine prices 17 million options per second and runs Monte Carlo across threads."

## Walk through each part (what, why, the number to quote)

**1. Black-Scholes and Greeks** (`black_scholes.py`)
- Price = S e^{−qT} N(d₁) − K e^{−rT} N(d₂). N(d₂) is the risk-neutral probability of finishing in the money; N(d₁) is the delta.
- Greeks are analytic, and tested against finite differences.
- I also check the PDE identity Θ + ½σ²S²Γ + rSΔ − rV = 0, which links theta and gamma.
- *Likely question:* "Why does the expected return μ not appear?" Delta hedging removes the risk, so the hedged portfolio must earn r; μ cancels in the derivation.

**2. Implied volatility** (`implied_vol.py`)
- Newton's method, using vega as the derivative, converges in a few steps.
- Far out of the money, vega is tiny, so Newton can jump out of range. I keep a bracket [lo, hi] and bisect whenever a Newton step would leave it, so it always converges.
- Prices outside no-arbitrage bounds return NaN because no volatility can produce them.

**3. Binomial tree** (`binomial.py`)
- CRR: u = e^{σ√Δt}, d = 1/u, risk-neutral p = (e^{(r−q)Δt} − d)/(u − d). Work backwards from the payoff.
- American: at each node take max(continue, exercise now).
- *Results:* the American put is 6.09 against 5.57 for the European; the American call equals the European call without dividends (a test checks it).
- *Likely question:* "Why does the tree error oscillate?" Whether the strike falls on a node or between nodes alternates with odd and even step counts. Fixes: average N and N+1 steps, or use Leisen-Reimer.

**4. Finite differences** (`pde.py`)
- Solve the Black-Scholes PDE in log-spot, where the coefficients are constant.
- Crank-Nicolson is second-order accurate and unconditionally stable, and each step is a tridiagonal solve, O(N).
- The kink of the payoff at the strike causes oscillations with Crank-Nicolson, so the first 4 steps are fully implicit (Rannacher smoothing).
- American options: after each step, take max(value, payoff).
- *Likely question:* "Why not explicit Euler?" It's only stable if Δt ≲ h²/σ², which forces tiny time steps.

**5. Monte Carlo** (`monte_carlo.py`)
- Simulate S_T exactly (no time steps needed for Europeans), average discounted payoffs, report the standard error. The error shrinks like 1/√n.
- **Antithetic variates:** pair z with −z.
- **Control variate:** use something with a known mean. For Europeans, the discounted stock (its mean is S₀ e^{−qT}). For Asians, the geometric Asian option, which has a closed form and is highly correlated with the arithmetic one. Optimal coefficient β = Cov/Var; variance shrinks by 1 − ρ². *Quote:* standard error divided by 36.
- **Barrier with Brownian bridge:** checking the barrier only at grid dates misses crossings in between, overpricing a knock-out by about 6% (9.21 against 8.67). The bridge gives the crossing probability exp(−2ab/(σ²Δt)) between two points, which removes the bias.
- **Pathwise delta:** differentiate the payoff path by path, 1{S_T > K} S_T/S₀.
- *Likely question:* "How do you know your Monte Carlo is right?" Tests check that the closed form lies inside the confidence interval, and over 100 seeds the normalised error has mean 0 and standard deviation about 1, so the error bars are honest.

**6. Delta hedging** (`hedging.py`)
- Sell a call at the price implied by σ_impl, hold Δ shares, rebalance N times, record the final P&L.
- Same vol: the average P&L is about 0, and the dispersion falls like 1/√N (16 times more rebalancing gives 4 times less error).
- Implied 25%, realised 18%: the seller earns 2.77 on average, matching the formula ∫ ½ e^{r(T−t)} Γ S² (σᵢ² − σᵣ²) dt (2.77).
- *Message:* an option desk is long or short volatility, not direction. A short-gamma book makes money when the market is calmer than priced.
- *Likely question:* "What about transaction costs?" The simulator has a cost parameter. More frequent rebalancing cuts hedging error but raises costs, so there's an optimal frequency (Leland's adjustment).

**7. Volatility surface** (`vol_surface.py`)
- Compute implied vols for each quote, then fit SVI per maturity: five parameters (level, slope, skew ρ, shift, curvature).
- Check the butterfly-arbitrage condition (the implied probability density must stay positive).
- **Be honest:** the included chain is synthetic. Run `scripts/fetch_chain.py SPY` on real data before your interviews and replace the figure. Real data brings wide bid-ask spreads on illiquid strikes and stale quotes, which you then have to filter.
- *Likely question:* "Why is there a skew?" Crash risk and demand for downside protection; volatility rises when markets fall.

**8. C++** (`cpp/`)
- Header-only, C++17, `std::erfc` for the normal CDF. Monte Carlo is split across `std::thread`s, each with its own seeded `mt19937_64`, so results are reproducible and threads share no state (no locks needed).
- Each thread accumulates the sum and sum of squares, giving the standard error at the end.
- *Likely questions:* "Why a separate RNG per thread?" A shared RNG would need locking (slow) and makes results depend on thread timing. "How would you call it from Python?" pybind11.

## Questions to prepare
1. Derive put-call parity. Show it holds in your tests.
2. Why is an American call on a non-dividend stock never exercised early?
3. Why does Crank-Nicolson need smoothing for options?
4. What is the convergence rate of Monte Carlo, and how did you improve the constant?
5. Explain the hedging P&L formula from Itô's lemma.
6. What are the limits of Black-Scholes, and how does the volatility surface fix them?
7. If you had another month, what would you add? (Heston, local volatility, Longstaff-Schwartz for American Monte Carlo.)

## CV line (use only once you understand everything above)
> **Option pricing engine** (Python, C++17): European, American and exotic options priced by Black-Scholes, binomial trees, Crank-Nicolson finite differences and Monte Carlo, cross-validated by 29 automated tests. Control variates reduced Monte Carlo error 36× on Asian options; Brownian-bridge correction removed a 6% barrier-pricing bias; delta-hedging simulator matches gamma-P&L theory within 0.2%; SVI volatility-surface calibration; multithreaded C++ Monte Carlo engine.

## Before you publish it
- Read every file and rerun everything yourself: `pytest`, `examples/make_figures.py`, and the C++ build.
- Fetch a real option chain and regenerate the volatility surface figure.
- Change anything you would have done differently: it should be your code, not a black box.
- Push it to GitHub with a clean history; put the link on your CV.
