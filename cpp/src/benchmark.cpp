// Benchmark: closed-form throughput and multithreaded Monte Carlo accuracy/speed.
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "pricer/black_scholes.hpp"
#include "pricer/monte_carlo.hpp"

using namespace pricer;
using clk = std::chrono::steady_clock;

int main() {
    const Market mkt{100.0, 0.05, 0.0};
    const Option call{100.0, 1.0, OptionType::Call};
    const double sigma = 0.2;
    const double exact = black_scholes(mkt, call, sigma).price;
    std::printf("Black-Scholes call (S=K=100, r=5%%, vol=20%%, T=1): %.6f\n\n", exact);

    // 1. Closed-form throughput: price + all Greeks for 10 million options.
    const int n = 10'000'000;
    double checksum = 0.0;
    auto t0 = clk::now();
    for (int i = 0; i < n; ++i) {
        const Option o{80.0 + 40.0 * (i % 1000) / 1000.0, 1.0, OptionType::Call};
        checksum += black_scholes(mkt, o, sigma).delta;
    }
    double secs = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("Closed form: %d options (price + 5 Greeks) in %.3f s -> %.1f M options/s (checksum %.1f)\n\n",
                n, secs, n / secs / 1e6, checksum);

    // 2. Monte Carlo: single thread vs all cores.
    const long long paths = 50'000'000;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned threads : {1u, cores}) {
        t0 = clk::now();
        const MCResult r = monte_carlo_european(mkt, call, sigma, paths, true, threads);
        secs = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("Monte Carlo, %u thread(s): %lld paths in %.3f s, price %.5f +/- %.5f (error vs exact %+.5f)\n",
                    threads, paths, secs, r.price, 1.96 * r.std_error, r.price - exact);
    }
    return 0;
}
