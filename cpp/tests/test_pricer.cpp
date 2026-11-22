// Minimal self-contained tests (no framework needed): run with ctest or ./test_pricer
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "pricer/black_scholes.hpp"
#include "pricer/monte_carlo.hpp"

using namespace pricer;

static int failures = 0;
#define CHECK_NEAR(a, b, tol)                                                              \
    do {                                                                                   \
        const double _a = (a), _b = (b);                                                   \
        if (std::fabs(_a - _b) > (tol)) {                                                  \
            std::printf("FAIL %s:%d  %s = %.10f, expected %.10f\n", __FILE__, __LINE__, #a, _a, _b); \
            ++failures;                                                                    \
        }                                                                                  \
    } while (0)

int main() {
    // Reference values (Hull): S=42, K=40, r=10%, vol=20%, T=0.5 -> call 4.76, put 0.81
    const Market hull{42.0, 0.10, 0.0};
    CHECK_NEAR(black_scholes(hull, {40.0, 0.5, OptionType::Call}, 0.2).price, 4.76, 5e-3);
    CHECK_NEAR(black_scholes(hull, {40.0, 0.5, OptionType::Put}, 0.2).price, 0.81, 5e-3);

    // Put-call parity with dividends
    const Market m{100.0, 0.05, 0.02};
    const Greeks c = black_scholes(m, {110.0, 2.0, OptionType::Call}, 0.25);
    const Greeks p = black_scholes(m, {110.0, 2.0, OptionType::Put}, 0.25);
    CHECK_NEAR(c.price - p.price, 100.0 * std::exp(-0.04) - 110.0 * std::exp(-0.1), 1e-12);
    CHECK_NEAR(c.delta - p.delta, std::exp(-0.04), 1e-12);

    // Delta and vega against finite differences
    const Option o{95.0, 1.0, OptionType::Call};
    const double h = 1e-5;
    const Market up{100.0 + h, 0.05, 0.0}, dn{100.0 - h, 0.05, 0.0}, mid{100.0, 0.05, 0.0};
    CHECK_NEAR(black_scholes(mid, o, 0.2).delta,
               (black_scholes(up, o, 0.2).price - black_scholes(dn, o, 0.2).price) / (2 * h), 1e-6);
    CHECK_NEAR(black_scholes(mid, o, 0.2).vega,
               (black_scholes(mid, o, 0.2 + h).price - black_scholes(mid, o, 0.2 - h).price) / (2 * h), 1e-5);

    // Implied vol round trip
    for (double vol : {0.08, 0.2, 0.45, 1.2}) {
        const double px = black_scholes(mid, {130.0, 0.75, OptionType::Call}, vol).price;
        CHECK_NEAR(implied_vol(mid, {130.0, 0.75, OptionType::Call}, px), vol, 1e-8);
    }

    // Monte Carlo within 4 standard errors of the closed form, for calls and puts
    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        const Option opt{100.0, 1.0, t};
        const MCResult r = monte_carlo_european(mid, opt, 0.2, 4'000'000, true, 4, 7);
        CHECK_NEAR(r.price, black_scholes(mid, opt, 0.2).price, 4 * r.std_error);
    }

    if (failures == 0) std::printf("All C++ tests passed.\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
