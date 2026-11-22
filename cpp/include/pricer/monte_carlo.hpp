// Multithreaded Monte Carlo pricer for European options under GBM (C++17).
// Each thread owns an independent, deterministically seeded RNG stream,
// so results are reproducible for a given (seed, n_threads).
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

#include "black_scholes.hpp"

namespace pricer {

struct MCResult {
    double price;
    double std_error;
    long long n_paths;
};

inline MCResult monte_carlo_european(const Market& m, const Option& o, double sigma,
                                     long long n_paths, bool antithetic = true,
                                     unsigned n_threads = std::thread::hardware_concurrency(),
                                     unsigned long long seed = 42) {
    n_threads = std::max(1u, n_threads);
    const double drift = (m.rate - m.dividend - 0.5 * sigma * sigma) * o.maturity;
    const double vol = sigma * std::sqrt(o.maturity);
    const double disc = std::exp(-m.rate * o.maturity);
    const double sign = (o.type == OptionType::Call) ? 1.0 : -1.0;

    // Each "sample" is one payoff (or the average of an antithetic pair).
    const long long n_samples = antithetic ? n_paths / 2 : n_paths;
    std::vector<double> sums(n_threads, 0.0), sums_sq(n_threads, 0.0);
    std::vector<std::thread> pool;

    auto worker = [&](unsigned id, long long begin, long long end) {
        std::mt19937_64 rng(seed + 1000003ULL * id);
        std::normal_distribution<double> normal(0.0, 1.0);
        double s = 0.0, s2 = 0.0;
        for (long long i = begin; i < end; ++i) {
            const double z = normal(rng);
            double pay = std::max(sign * (m.spot * std::exp(drift + vol * z) - o.strike), 0.0);
            if (antithetic) {
                const double pay2 = std::max(sign * (m.spot * std::exp(drift - vol * z) - o.strike), 0.0);
                pay = 0.5 * (pay + pay2);
            }
            s += pay;
            s2 += pay * pay;
        }
        sums[id] = s;
        sums_sq[id] = s2;
    };

    const long long chunk = n_samples / n_threads;
    for (unsigned t = 0; t < n_threads; ++t) {
        const long long begin = t * chunk;
        const long long end = (t + 1 == n_threads) ? n_samples : begin + chunk;
        pool.emplace_back(worker, t, begin, end);
    }
    for (auto& th : pool) th.join();

    double s = 0.0, s2 = 0.0;
    for (unsigned t = 0; t < n_threads; ++t) { s += sums[t]; s2 += sums_sq[t]; }
    const double mean = s / n_samples;
    const double var = (s2 / n_samples - mean * mean) * n_samples / (n_samples - 1);
    return {disc * mean, disc * std::sqrt(var / n_samples), n_paths};
}

}  // namespace pricer
