/**
 * @file bs_call_price.h
 * @brief Compact Black-Scholes helpers + call price.
 *
 * Exposes:
 *  - Phi_real(z): standard normal CDF Phi(z).
 *  - phi(z):      standard normal PDF phi(z).
 *  - bs_price_call(S,K,r,q,sigma,T): European call price (with continuous yield q).
 *
 * Intended as the minimal building block for Greeks.
 */

#pragma once
#include <algorithm>
#include <cmath>

// Phi(z): standard normal CDF
inline double Phi_real(double z) {
    static constexpr double INV_SQRT_2 = 0.70710678118654752440; // 1/sqrt(2)
    return 0.5 * std::erfc(-z * INV_SQRT_2);
}

// phi(z): standard normal PDF
inline double phi(double z) {
    static constexpr double INV_SQRT_2PI = 0.39894228040143267794; // 1/sqrt(2*pi)
    return INV_SQRT_2PI * std::exp(-0.5 * z * z);
}

// Black-Scholes call-price
inline double bs_price_call(double S, double K, double r, double q, double sigma, double T) {
    const double DF     = std::exp(-r * T);
    const double F      = S * std::exp((r - q) * T);
    const double sigmaT = sigma * std::sqrt(std::max(T, 0.0));
    if (sigmaT == 0.0) return DF * std::max(F - K, 0.0);

    double ln_F_over_K;
    if (K > 0.0) { 
        const double x = (F - K) / K;
        ln_F_over_K = (std::abs(x) <= 1e-12) ? std::log1p(x) : std::log(F / K);
    } else {
        ln_F_over_K = std::log(F / K);
    }

    const double d1 = (ln_F_over_K + 0.5 * sigma * sigma * T) / sigmaT;
    const double d2 = d1 - sigmaT;

    return DF * (F * Phi_real(d1) - K * Phi_real(d2));
}

