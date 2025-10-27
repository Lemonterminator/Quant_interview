#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>

namespace quant {

class InverseCumulativeNormal {
  public:
    explicit InverseCumulativeNormal(double average = 0.0, double sigma = 1.0)
    : average_(average), sigma_(sigma) {}

    // Scalar call: return average + sigma * Φ^{-1}(x)
    inline double operator()(double x) const {
        return average_ + sigma_ * standard_value(x);
    }

    // Vector overload: out[i] = average + sigma * Φ^{-1}(in[i]) for i in [0, n)
    inline void operator()(const double* in, double* out, std::size_t n) const {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = average_ + sigma_ * standard_value(in[i]);
        }
    }

    // Standardized value: inverse CDF with average=0, sigma=1.
    // Baseline: deliberately crude but correct bisection. Replace internals with your faster method.
    static inline double standard_value(double x) {
        // Handle edge and invalid cases defensively.
        if (x <= 0.0) return -std::numeric_limits<double>::infinity();
        if (x >= 1.0) return  std::numeric_limits<double>::infinity();

        // Piecewise structure left in place so you can drop in rational approximations.
        if (x < x_low_ || x > x_high_) {
            double z = tail_value_baseline(x);   // << replace with tail mapping + rational
        #ifdef ICN_ENABLE_HALLEY_REFINEMENT
            z = halley_refine(z, x);
        #endif
            return z;
        } else {
            double z = central_value_baseline(x); // << replace with central-region rational
        #ifdef ICN_ENABLE_HALLEY_REFINEMENT
            z = halley_refine(z, x);
        #endif
            return z;
        }
    }

  private:
    // ---- Baseline numerics (intentionally slow but stable) ------------------

    // Standard normal pdf
    static inline double phi(double z) {
        // 1/sqrt(2π) * exp(-z^2 / 2)
        constexpr double INV_SQRT_2PI =
            0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
        return INV_SQRT_2PI * std::exp(-0.5 * z * z);
    }

    // Standard normal cdf using erfc: Φ(z) = 0.5 * erfc(-z/√2)
    static inline double Phi(double z) {
        constexpr double INV_SQRT_2 =
            0.707106781186547524400844362104849039284835937688474036588; // 1/√2
        return 0.5 * std::erfc(-z * INV_SQRT_2);
    }

    // Crude but reliable invert via bisection; brackets wide enough for double tails.
    static inline double invert_bisect(double x) {
        // Monotone Φ(z); find z with Φ(z)=x.
        double lo = -12.0;
        double hi =  12.0;
        // Tighten bracket using symmetry for speed (optional micro-optimization).
        if (x < 0.5) {
            hi = 0.0;
        } else {
            lo = 0.0;
        }

        // Bisection iterations: ~60 is enough for double precision on this interval.
        for (int iter = 0; iter < 80; ++iter) {
            double mid = 0.5 * (lo + hi);
            double cdf = Phi(mid);
            if (cdf < x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    // Baseline central-region value: currently just bisection.
    static inline double central_value_baseline(double x) {
        // TODO(candidate): Replace with rational approximation around x≈0.5
        return invert_bisect(x);
    }

    // Baseline tail handler: currently just bisection (slow for extreme x).
    static inline double tail_value_baseline(double x) {
        // TODO(candidate): Implement tail mapping t = sqrt(-2*log(m)) with rational in t
        return invert_bisect(x);
    }

#ifdef ICN_ENABLE_HALLEY_REFINEMENT
    // One-step Halley refinement (3rd order). Usually brings result to full double precision.
    static inline double halley_refine(double z, double x) {
        // r = (Φ(z) - x) / φ(z)
        const double f = Phi(z);
        const double p = phi(z);
        const double r = (f - x) / std::max(p, std::numeric_limits<double>::min());
        // Halley: z_{new} = z - r / (1 - 0.5*z*r)
        const double denom = 1.0 - 0.5 * z * r;
        return z - r / (denom != 0.0 ? denom
                                     : std::copysign(std::numeric_limits<double>::infinity(), denom));
    }
#endif

    // ---- State & constants ---------------------------------------------------

    double average_, sigma_;

    // Region split (you may adjust in your improved version).
    static constexpr double x_low_  = 0.02425;         // ~ Φ(-2.0)
    static constexpr double x_high_ = 1.0 - x_low_;
};

} // namespace quant

/*
Minimal usage example (not part of API, kept here for convenience):

#include <iostream>
#include <array>

int main() {
    // --- Scalar usage ---
    quant::InverseCumulativeNormal icn; // mean=0, sigma=1
    double xs[] = {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1-1e-6, 1-1e-12};
    for (double x : xs) {
        double z = icn(x); // z = Φ^{-1}(x)
        std::cout << "scalar  x=" << x << "  z=" << z << "\n";
    }

    // --- Vector/array usage (multiple values at once) ---
    const double xin[] = {0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999};
    double zout[std::size(xin)];
    icn(xin, zout, std::size(xin)); // out[i] = Φ^{-1}(xin[i])

    for (std::size_t i = 0; i < std::size(xin); ++i) {
        std::cout << "vector  x=" << xin[i] << "  z=" << zout[i] << "\n";
    }

    return 0;
}
*/
