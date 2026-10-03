// Regression test: scalar / vector / parallel Gamma must agree on degenerate inputs.
//
// Previously the scalar Gamma guarded with `T < 0 || S < 0 || sigma < 0`, so
// sigma == 0, T == 0 or S == 0 slipped through and produced NaN (0/0), while
// Gamma_par returned 0.0. All three entry points must now return 0.0 there.

#include <cmath>
#include <iostream>
#include <vector>

#include "black_scholes.h"

namespace bs = black_scholes;

int main() {
    struct Case { const char* name; double S, K, r, q, sigma, T; };
    const std::vector<Case> degenerate = {
        {"sigma == 0",        100.0, 100.0, 0.0, 0.0, 0.0,  1.0},
        {"T == 0",            100.0, 100.0, 0.0, 0.0, 0.2,  0.0},
        {"S == 0",              0.0, 100.0, 0.0, 0.0, 0.2,  1.0},
        {"sigma == 0 and T == 0", 100.0, 100.0, 0.0, 0.0, 0.0, 0.0},
        {"T < 0",             100.0, 100.0, 0.0, 0.0, 0.2, -1.0},
        {"sigma < 0",         100.0, 100.0, 0.0, 0.0, -0.2, 1.0},
        {"S < 0",            -100.0, 100.0, 0.0, 0.0, 0.2,  1.0},
    };

    int failures = 0;
    for (const auto& c : degenerate) {
        const double scalar = bs::Gamma(c.S, c.K, c.r, c.q, c.sigma, c.T);

        const std::vector<double> S{c.S}, K{c.K}, r{c.r}, q{c.q}, sigma{c.sigma}, T{c.T};
        std::vector<double> vec, par;
        bs::Gamma(S, K, r, q, sigma, T, vec);
        bs::Gamma_par(S, K, r, q, sigma, T, par);

        const bool ok = scalar == 0.0 && vec[0] == 0.0 && par[0] == 0.0;
        if (!ok) {
            std::cerr << "FAIL [" << c.name << "]: scalar=" << scalar
                      << " vector=" << vec[0] << " parallel=" << par[0]
                      << " (all expected 0)\n";
            ++failures;
        }
    }

    // Sanity: a regular input is untouched by the guard change.
    const double g = bs::Gamma(100.0, 100.0, 0.0, 0.0, 0.2, 1.0);
    if (!std::isfinite(g) || std::abs(g - 0.019847627373850) > 1e-12) {
        std::cerr << "FAIL [regular ATM gamma]: got " << g << "\n";
        ++failures;
    }

    if (failures == 0) {
        std::cout << "gamma_edge_test: all tests passed." << std::endl;
        return 0;
    }
    std::cerr << "gamma_edge_test: " << failures << " failure(s)." << std::endl;
    return 1;
}
