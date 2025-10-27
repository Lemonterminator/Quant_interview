#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include "bs_call_price.h"

namespace {
    bool approx_equal(double a, double b, double tol = 1e-12) {
        double diff = std::abs(a - b);
        double scale = std::max({1.0, std::abs(a), std::abs(b)});
        return diff <= tol * scale;
    }
}

int main() {
    int failures = 0;

    // Known values (from standard normal CDF)
    struct Case { double z; double expected; double tol; };
    std::vector<Case> cases = {
        {  0.0, 0.5,                 1e-15 },
        {  1.0, 0.8413447460685429,  1e-15 },
        { -1.0, 0.15865525393145707, 1e-15 },
        {  2.0, 0.9772498680518208,  1e-15 },
        { -2.0, 0.02275013194817921, 1e-15 },
        {  3.0, 0.9986501019683699,  1e-15 },
        { -3.0, 0.0013498980316301035, 1e-15 }
    };

    for (const auto& c : cases) {
        double v = Phi_real(c.z);
        if (!approx_equal(v, c.expected, c.tol)) {
            std::cerr << std::setprecision(17)
                      << "Phi_real(" << c.z << ") got " << v
                      << ", expected " << c.expected << "\n";
            ++failures;
        }
    }

    // Symmetry check: Phi(z) = 1 - Phi(-z)
    for (double z : { -4.0, -1.2345, -0.1, 0.0, 0.1, 1.2345, 4.0 }) {
        double lhs = Phi_real(z);
        double rhs = 1.0 - Phi_real(-z);
        if (!approx_equal(lhs, rhs, 1e-15)) {
            std::cerr << std::setprecision(17)
                      << "Symmetry failed at z=" << z
                      << ": Phi(z)=" << lhs << ", 1-Phi(-z)=" << rhs << "\n";
            ++failures;
        }
    }

    // Monotonicity check over a grid
    double prev = -std::numeric_limits<double>::infinity();
    for (int i = -50; i <= 50; ++i) {
        double z = 0.2 * i; // from -10 to 10
        double v = Phi_real(z);
        if (!(v >= prev)) {
            std::cerr << "Monotonicity failed between z=" << (z - 0.2)
                      << " and z=" << z << "\n";
            ++failures;
        }
        prev = v;
    }

    // Extreme tails
    if (!(Phi_real(10.0) > 0.9999999999)) { ++failures; std::cerr << "Tail too low at z=10\n"; }
    if (!(Phi_real(-10.0) < 1e-10))       { ++failures; std::cerr << "Tail too high at z=-10\n"; }

    if (failures == 0) {
        std::cout << "Phi_real_test: all tests passed." << std::endl;
        return 0;
    } else {
        std::cerr << "Phi_real_test: " << failures << " failure(s)." << std::endl;
        return 1;
    }
}
