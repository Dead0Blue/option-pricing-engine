// Header-only Black-Scholes-Merton pricer and Greeks (C++17).
#pragma once

#include <cmath>
#include <stdexcept>

namespace pricer {

enum class OptionType { Call, Put };

struct Market {
    double spot;
    double rate;      // continuously compounded risk-free rate
    double dividend;  // continuous dividend yield
};

struct Option {
    double strike;
    double maturity;  // years
    OptionType type;
};

struct Greeks {
    double price, delta, gamma, vega, theta, rho;
};

inline double norm_pdf(double x) { return 0.3989422804014327 * std::exp(-0.5 * x * x); }
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865476); }

inline Greeks black_scholes(const Market& m, const Option& o, double sigma) {
    if (sigma <= 0.0 || o.maturity <= 0.0 || m.spot <= 0.0 || o.strike <= 0.0)
        throw std::invalid_argument("black_scholes: inputs must be positive");
    const double sqrt_t = std::sqrt(o.maturity);
    const double vol_sqrt_t = sigma * sqrt_t;
    const double d1 = (std::log(m.spot / o.strike) +
                       (m.rate - m.dividend + 0.5 * sigma * sigma) * o.maturity) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;
    const double df_r = std::exp(-m.rate * o.maturity);
    const double df_q = std::exp(-m.dividend * o.maturity);
    const double pdf = norm_pdf(d1);

    Greeks g{};
    g.gamma = df_q * pdf / (m.spot * vol_sqrt_t);
    g.vega = m.spot * df_q * pdf * sqrt_t;
    const double decay = -m.spot * df_q * pdf * sigma / (2.0 * sqrt_t);
    if (o.type == OptionType::Call) {
        g.price = m.spot * df_q * norm_cdf(d1) - o.strike * df_r * norm_cdf(d2);
        g.delta = df_q * norm_cdf(d1);
        g.theta = decay - m.rate * o.strike * df_r * norm_cdf(d2) + m.dividend * m.spot * df_q * norm_cdf(d1);
        g.rho = o.strike * o.maturity * df_r * norm_cdf(d2);
    } else {
        g.price = o.strike * df_r * norm_cdf(-d2) - m.spot * df_q * norm_cdf(-d1);
        g.delta = df_q * (norm_cdf(d1) - 1.0);
        g.theta = decay + m.rate * o.strike * df_r * norm_cdf(-d2) - m.dividend * m.spot * df_q * norm_cdf(-d1);
        g.rho = -o.strike * o.maturity * df_r * norm_cdf(-d2);
    }
    return g;
}

// Safeguarded Newton implied volatility (bisection fallback). Returns NaN if no solution.
inline double implied_vol(const Market& m, const Option& o, double target, double tol = 1e-10) {
    double lo = 1e-6, hi = 5.0, sigma = 0.2;
    if (target <= black_scholes(m, o, lo).price || target >= black_scholes(m, o, hi).price)
        return std::nan("");
    for (int i = 0; i < 100; ++i) {
        const Greeks g = black_scholes(m, o, sigma);
        const double diff = g.price - target;
        if (std::fabs(diff) < tol) break;
        (diff < 0 ? lo : hi) = sigma;
        const double newton = sigma - diff / g.vega;
        sigma = (newton > lo && newton < hi) ? newton : 0.5 * (lo + hi);
    }
    return sigma;
}

}  // namespace pricer
