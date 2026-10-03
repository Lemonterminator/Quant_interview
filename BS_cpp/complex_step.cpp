#include "complex_step.h"
#include "black_scholes.h"
#include <cmath>
#include <stdexcept>

namespace black_scholes {
namespace {
void validate_order(int order) {
    if (order < 2 || order > 6) throw std::invalid_argument("order must be 2, 3, 4, 5, 6");
}
void validate_step(double h) {
    if (!std::isfinite(h) || h <= 0) throw std::invalid_argument("h must be positive and finite");
}
double finite_result(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("Unrepresentable complex-step Greek");
    return x;
}
}

std::complex<double> Phi_t(std::complex<double> z, int order) {
    validate_order(order);
    const double a = z.real(), b = z.imag();
    if (!std::isfinite(a) || !std::isfinite(b)) throw std::invalid_argument("CDF input must be finite");
    if (b == 0) return {Phi_real(a), 0};
    const double pdf = 0.39894228040143267794 * std::exp(-0.5 * a * a);
    const double a2 = a * a, b2 = b * b;
    double real = Phi_real(a) + 0.5 * a * b2 * pdf;
    double imag = b * pdf;
    if (order >= 3) imag += (1 - a2) / 6 * b * b2 * pdf;
    if (order >= 4) real += (3 - a2) * a / 24 * b2 * b2 * pdf;
    if (order >= 5) imag += ((a2 - 6) * a2 + 3) / 120 * b * b2 * b2 * pdf;
    if (order >= 6) real += ((a2 - 10) * a2 + 15) * a / 720 * b2 * b2 * b2 * pdf;
    return {real, imag};
}

std::complex<double> bs_price_call_complex(std::complex<double> S, double K,
    double r, double q, double sigma, double T, int order) {
    validate_order(order);
    if (!std::isfinite(S.real()) || !std::isfinite(S.imag()) || S.real() <= 0 ||
        !std::isfinite(K) || K <= 0 || !std::isfinite(r) || !std::isfinite(q) ||
        !std::isfinite(sigma) || sigma <= 0 || !std::isfinite(T) || T <= 0)
        throw std::invalid_argument("Complex pricing requires finite S.real(),K,sigma,T>0 and finite r,q");
    const double tau = sigma * std::sqrt(T);
    if (!std::isfinite(tau) || tau <= 0) throw std::invalid_argument("Invalid sigma*sqrt(T)");
    const auto F = S * std::exp((r - q) * T);
    const auto d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / tau;
    const auto value = std::exp(-r * T) * (F * Phi_t(d1, order) - K * Phi_t(d1 - tau, order));
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::overflow_error("Unrepresentable complex call price");
    return value;
}

double delta_complex_step(double S, double K, double r, double q,
    double sigma, double T, double h, int order) {
    validate_step(h);
    return finite_result(bs_price_call_complex({S, h}, K, r, q, sigma, T, order).imag() / h);
}

double gamma_cs_real(double S, double K, double r, double q,
    double sigma, double T, double h, int order) {
    validate_step(h);
    const auto complex_price = bs_price_call_complex({S, h}, K, r, q, sigma, T, order);
    return finite_result(-2 * (complex_price.real() - bs_price_call(S, K, r, q, sigma, T)) / (h * h));
}

double gamma_45_deg(double S, double K, double r, double q,
    double sigma, double T, double h, int order) {
    validate_step(h);
    const std::complex<double> omega(1 / std::sqrt(2.), 1 / std::sqrt(2.));
    const auto cf = bs_price_call_complex(S + omega * h, K, r, q, sigma, T, order);
    const auto cb = bs_price_call_complex(S - omega * h, K, r, q, sigma, T, order);
    return finite_result((cf + cb).imag() / (h * h));
}
} // namespace black_scholes
