/**
 * @file black_scholes.h
 * @brief Compact helpers and all numerical routines for Black-Scholes model. 
 *
 * Exposes:
 *  - Phi_real(z): standard normal CDF Phi(z).
 *  - phi(z):      standard normal PDF phi(z).
 *
 * Intended as the minimal building block for Greeks.
 */



#pragma once

#include <vector>
#include <algorithm>    // std::transform
#include <cmath>        // std::erfc
#include <execution>    // C++17 execution policies (optional)
#include <limits>       // std::numeric_limits
#include <stdexcept>    // std::invalid_argument

namespace black_scholes {
    constexpr double INV_SQRT_2 = 0.70710678118654752440;
    constexpr double INV_SQRT_2PI = 0.39894228040143267794;
    constexpr double LOG_SQRT_2PI = 0.91893853320467274178;

    inline double Phi_real(double z) noexcept {
        constexpr double INV_SQRT_2 = 0.70710678118654752440;
        return 0.5 * std::erfc(-z * INV_SQRT_2);
    }

    
    // Parallel + vectorization friendly 
    inline void Phi_real_par(const std::vector<double>& z, std::vector<double>& out) {
        const auto* const out_as_const = static_cast<const std::vector<double>*>(&out);
        if (&z == out_as_const) {
            std::transform(std::execution::par_unseq, out.begin(), out.end(), out.begin(),
                           [](double v) { return Phi_real(v); });
            return;
        }

        out.resize(z.size());
        std::transform(std::execution::par_unseq, z.begin(), z.end(), out.begin(),
                       [](double v) { return Phi_real(v); });
    }

    namespace detail {
        inline double log_F_over_K(double F, double K) noexcept {
            if (K > 0.0) {
                const double x = (F - K) / K;
                if (std::abs(x) <= 1e-12) {
                    return std::log1p(x);
                }
            }
            return std::log(F / K);
        }

        inline double safe_ratio(double numerator, double denom) noexcept {
            if (denom > 0.0) {
                return numerator / denom;
            }
            if (numerator > 0.0) {
                return std::numeric_limits<double>::infinity();
            }
            if (numerator < 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            return 0.0;
        }

        inline double clamp_sqrt_time(double T) noexcept {
            return std::sqrt(std::max(T, 0.0));
        }

        inline void validate_size(std::size_t expected, std::size_t actual) {
            if (actual != expected) {
                throw std::invalid_argument("delta_call expects all input vectors to share the same length.");
            }
        }
    } // namespace detail

    inline double delta_call(double S, double K, double r, double q, double sigma, double T) {
        const double F = S * std::exp((r - q) * T);
        const double ln_F_over_K = detail::log_F_over_K(F, K);
        const double numerator = ln_F_over_K + 0.5 * sigma * sigma * T;
        const double denom = sigma * detail::clamp_sqrt_time(T);
        const double d1 = detail::safe_ratio(numerator, denom);
        return std::exp(-q * T) * Phi_real(d1);
    }

    inline void delta_call(const std::vector<double>& S,
                           const std::vector<double>& K,
                           const std::vector<double>& r,
                           const std::vector<double>& q,
                           const std::vector<double>& sigma,
                           const std::vector<double>& T,
                           std::vector<double>& out) {
        const std::size_t n = S.size();
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());

        out.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = delta_call(S[i], K[i], r[i], q[i], sigma[i], T[i]);
        }
    }

    inline void delta_call_par(const std::vector<double>& S,
                               const std::vector<double>& K,
                               const std::vector<double>& r,
                               const std::vector<double>& q,
                               const std::vector<double>& sigma,
                               const std::vector<double>& T,
                               std::vector<double>& out) {
        const std::size_t n = S.size();
        if (n == 0) {
            out.clear();
            return;
        }
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());

        out.resize(n);
        std::vector<double> d1(n);

        auto* const d1_data = d1.data();
        auto* const discount_data = out.data();
        const double* const S_data = S.data();
        const double* const K_data = K.data();
        const double* const r_data = r.data();
        const double* const q_data = q.data();
        const double* const sigma_data = sigma.data();
        const double* const T_data = T.data();

        std::for_each(std::execution::par_unseq, d1.begin(), d1.end(),
                      [&](double& slot) {
                          const std::size_t idx = static_cast<std::size_t>(&slot - d1_data);
                          const double t = T_data[idx];
                          const double F = S_data[idx] * std::exp((r_data[idx] - q_data[idx]) * t);
                          const double log_fk = detail::log_F_over_K(F, K_data[idx]);
                          const double sigma_val = sigma_data[idx];
                          const double numerator = log_fk + 0.5 * sigma_val * sigma_val * t;
                          const double denom = sigma_val * detail::clamp_sqrt_time(t);
                          slot = detail::safe_ratio(numerator, denom);
                          discount_data[idx] = std::exp(-q_data[idx] * t);
                      });

        Phi_real_par(d1, d1);
        std::transform(std::execution::par_unseq, out.begin(), out.end(), d1.begin(), out.begin(),
                       [](double discount, double phi_val) { return discount * phi_val; });
    }


    inline double Gamma(double S, double K, double r, double q, double sigma, double T){
        if (T < 0 || S <0 || sigma < 0) {
            return 0.0;
        }
        double sT =  sigma * std::sqrt(T);
        double F = S * std::exp((r-q)*T);
        double ln = detail::log_F_over_K(F, K);
        double d1 = (ln + 0.5 *sigma * sigma* T) / sT;
        double log_phi_d1 = -0.5 * d1*d1 - LOG_SQRT_2PI;
        double phi_d1 = std::exp(log_phi_d1);
        double out = std::exp(-q * T) * phi_d1 / (S * sT);
        return out;
    }

    inline void Gamma(const std::vector<double>& S,
                           const std::vector<double>& K,
                           const std::vector<double>& r,
                           const std::vector<double>& q,
                           const std::vector<double>& sigma,
                           const std::vector<double>& T,
                           std::vector<double>& out){

        const std::size_t n = S.size();
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());
        out.resize(n);

        for (std::size_t i = 0; i < n; ++i){
            out[i] = Gamma(S[i], K[i], r[i], q[i], sigma[i], T[i]);
        }
    }

    inline void Gamma_par(const std::vector<double>& S,
                               const std::vector<double>& K,
                               const std::vector<double>& r,
                               const std::vector<double>& q,
                               const std::vector<double>& sigma,
                               const std::vector<double>& T,
                               std::vector<double>& out){

        const std::size_t n = S.size();
        if (n == 0) {
            out.clear();
            return;
        }
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());

        out.assign(n, 0.0);

        auto* const out_data = out.data();
        const double* const S_data = S.data();
        const double* const K_data = K.data();
        const double* const r_data = r.data();
        const double* const q_data = q.data();
        const double* const sigma_data = sigma.data();
        const double* const T_data = T.data();

        // Matches Python_prototype/black_scholes.Gamma for speed + numerical safety.
        std::for_each(std::execution::par_unseq, out.begin(), out.end(),
                      [&](double& slot) {
                          const std::size_t idx = static_cast<std::size_t>(&slot - out_data);
                          const double S_val = S_data[idx];
                          const double sigma_val = sigma_data[idx];
                          const double t = std::max(T_data[idx], 0.0);
                          if (S_val <= 0.0 || sigma_val <= 0.0 || t <= 0.0) {
                              slot = 0.0;
                              return;
                          }

                          const double sqrt_t = detail::clamp_sqrt_time(t);
                          const double sT = sigma_val * sqrt_t;
                          if (sT <= 0.0) {
                              slot = 0.0;
                              return;
                          }

                          const double F = S_val * std::exp((r_data[idx] - q_data[idx]) * t);
                          const double log_fk = detail::log_F_over_K(F, K_data[idx]);
                          const double d1_num = log_fk + 0.5 * sigma_val * sigma_val * t;
                          const double d1 = detail::safe_ratio(d1_num, sT);
                          const double log_phi_d1 = -0.5 * d1 * d1 - LOG_SQRT_2PI;
                          const double phi_d1 = std::exp(log_phi_d1);

                          const double discount = std::exp(-q_data[idx] * t);
                          const double denom = S_val * sT;
                          slot = detail::safe_ratio(discount * phi_d1, denom);
                      });

                            
    }

    // TODO: Implement all other missing functions from prototype black_scholes.py in blackscholes.h
    // As in the same style, first the numeric function for a single value, then parallelize with another funciton _par
    inline double bs_price_call(double S, double K, double r, double q, double sigma, double T){
        double DF = std::exp(-r*T);

        double F =  std::exp((r-q)*T);

        double sT = sigma * std::sqrt(std::max(T, 0.0));

        if (sT == 0.0) {return 0.0;}
        

        double ln_F_over_K = detail::log_F_over_K(F, K);

        double d1 = (ln_F_over_K + 0.5 * sigma * sigma * T) / sT;
        double d2 = d1 - sT;

        double out = DF * (F * Phi_real(d1) - K * Phi_real(d2));

        return out;
    }

    inline void bs_price_call(const std::vector<double>& S,
                           const std::vector<double>& K,
                           const std::vector<double>& r,
                           const std::vector<double>& q,
                           const std::vector<double>& sigma,
                           const std::vector<double>& T,
                           std::vector<double>& out){

        const std::size_t n = S.size();
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());
        out.resize(n);

        for (std::size_t i = 0; i < n; ++i){
            out[i] = bs_price_call(S[i], K[i], r[i], q[i], sigma[i], T[i]);
        }
    }


    inline void bs_price_call_par(const std::vector<double>& S,
                                const std::vector<double>& K,
                                const std::vector<double>& r,
                                const std::vector<double>& q,
                                const std::vector<double>& sigma, 
                                const std::vector<double>& T, 
                                std::vector<double>& out){
        const std::size_t n = S.size();
        if (n == 0) {
            out.clear();
            return;
        }
        detail::validate_size(n, K.size());
        detail::validate_size(n, r.size());
        detail::validate_size(n, q.size());
        detail::validate_size(n, sigma.size());
        detail::validate_size(n, T.size());

        out.assign(n, 0.0);

        auto* const out_data = out.data();
        const double* const S_data = S.data();
        const double* const K_data = K.data();
        const double* const r_data = r.data();
        const double* const q_data = q.data();
        const double* const sigma_data = sigma.data();
        const double* const T_data = T.data();
        
        std::for_each(std::execution::par_unseq, out.begin(), out.end()
            [&](double& slot) {

                
            }
        );
    }
}
