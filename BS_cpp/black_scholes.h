/**
 * @file black_scholes.h
 * @brief Declarations for Black-Scholes analytics and helpers.
 */

#pragma once

#include <vector>

namespace black_scholes {

// Standard normal CDF Phi(z).
// Returns Phi(z) as a double.
double Phi_real(double z) noexcept;

// Vectorized Phi(z) applying Phi_real element-wise.
// Writes results to `out` (resized as needed).
void Phi_real_par(const std::vector<double>& z, std::vector<double>& out);

// Black-Scholes delta (call) for a single set of inputs.
// Returns delta as a double.
double delta_call(double S, double K, double r, double q, double sigma, double T);

// Vectorized Black-Scholes delta (call).
// Writes element-wise deltas to `out` (resized as needed).
void delta_call(const std::vector<double>& S,
                const std::vector<double>& K,
                const std::vector<double>& r,
                const std::vector<double>& q,
                const std::vector<double>& sigma,
                const std::vector<double>& T,
                std::vector<double>& out);

// Parallel vectorized Black-Scholes delta (call).
void delta_call_par(const std::vector<double>& S,
                    const std::vector<double>& K,
                    const std::vector<double>& r,
                    const std::vector<double>& q,
                    const std::vector<double>& sigma,
                    const std::vector<double>& T,
                    std::vector<double>& out);

// Black-Scholes gamma (call) for a single set of inputs.
// Returns gamma as a double.
double Gamma(double S, double K, double r, double q, double sigma, double T);

// Vectorized Black-Scholes gamma (call).
void Gamma(const std::vector<double>& S,
           const std::vector<double>& K,
           const std::vector<double>& r,
           const std::vector<double>& q,
           const std::vector<double>& sigma,
           const std::vector<double>& T,
           std::vector<double>& out);

// Parallel vectorized Black-Scholes gamma (call).
void Gamma_par(const std::vector<double>& S,
               const std::vector<double>& K,
               const std::vector<double>& r,
               const std::vector<double>& q,
               const std::vector<double>& sigma,
               const std::vector<double>& T,
               std::vector<double>& out);

// Black-Scholes call price for a single set of inputs.
// Returns price as a double.
double bs_price_call(double S, double K, double r, double q, double sigma, double T);

// Vectorized Black-Scholes call price.
void bs_price_call(const std::vector<double>& S,
                   const std::vector<double>& K,
                   const std::vector<double>& r,
                   const std::vector<double>& q,
                   const std::vector<double>& sigma,
                   const std::vector<double>& T,
                   std::vector<double>& out);

// Parallel vectorized Black-Scholes call price.
void bs_price_call_par(const std::vector<double>& S,
                       const std::vector<double>& K,
                       const std::vector<double>& r,
                       const std::vector<double>& q,
                       const std::vector<double>& sigma,
                       const std::vector<double>& T,
                       std::vector<double>& out);

// Forward finite-difference delta (scalar).
// Returns (C(S+h)-C(S))/h as a double.
double delta_forward(double S, double K, double r, double q, double sigma, double T, double h = 1e-5);

// Forward finite-difference delta (vectorized).
void delta_forward(const std::vector<double>& S,
                   const std::vector<double>& K,
                   const std::vector<double>& r,
                   const std::vector<double>& q,
                   const std::vector<double>& sigma,
                   const std::vector<double>& T,
                   std::vector<double>& out,
                   double h = 1e-5);

// Forward finite-difference delta (parallel vectorized).
void delta_forward_par(const std::vector<double>& S,
                       const std::vector<double>& K,
                       const std::vector<double>& r,
                       const std::vector<double>& q,
                       const std::vector<double>& sigma,
                       const std::vector<double>& T,
                       std::vector<double>& out,
                       double h = 1e-5);

// Forward finite-difference gamma (scalar).
// Returns (C(S+2h)-2C(S+h)+C(S))/h^2 as a double.
double gamma_forward(double S, double K, double r, double q, double sigma, double T, double h = 1e-5);

// Forward finite-difference gamma (vectorized).
void gamma_forward(const std::vector<double>& S,
                   const std::vector<double>& K,
                   const std::vector<double>& r,
                   const std::vector<double>& q,
                   const std::vector<double>& sigma,
                   const std::vector<double>& T,
                   std::vector<double>& out,
                   double h = 1e-5);

// Forward finite-difference gamma (parallel vectorized).
void gamma_forward_par(const std::vector<double>& S,
                       const std::vector<double>& K,
                       const std::vector<double>& r,
                       const std::vector<double>& q,
                       const std::vector<double>& sigma,
                       const std::vector<double>& T,
                       std::vector<double>& out,
                       double h = 1e-5);

} // namespace black_scholes

