#pragma once
#include <complex>

namespace black_scholes {
// order is the highest retained imaginary Taylor power, not derivative order.
// This local expansion requires a small imaginary part; P3 is the default.
std::complex<double> Phi_t(std::complex<double> z, int order = 3);
std::complex<double> bs_price_call_complex(std::complex<double> S, double K,
    double r, double q, double sigma, double T, int order = 3);
double delta_complex_step(double S, double K, double r, double q,
    double sigma, double T, double h = 1e-20, int order = 3);
double gamma_cs_real(double S, double K, double r, double q,
    double sigma, double T, double h = 1e-5, int order = 3);
double gamma_45_deg(double S, double K, double r, double q,
    double sigma, double T, double h = 1e-5, int order = 3);
} // namespace black_scholes
