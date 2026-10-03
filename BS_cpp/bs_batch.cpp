#include "bs_batch.h"

#include "black_scholes.h"
#include "complex_step.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace black_scholes {
namespace {

void validate(const CallInputs& input) {
    const auto n = input.S.size();
    if (input.K.size() != n || input.r.size() != n || input.q.size() != n ||
        input.sigma.size() != n || input.T.size() != n) {
        throw std::invalid_argument("All batch inputs must have the same length");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(input.S[i]) || !std::isfinite(input.K[i]) ||
            !std::isfinite(input.r[i]) || !std::isfinite(input.q[i]) ||
            !std::isfinite(input.sigma[i]) || !std::isfinite(input.T[i]) ||
            input.S[i] <= 0.0 || input.K[i] <= 0.0 ||
            input.sigma[i] < 0.0 || input.T[i] < 0.0) {
            throw std::invalid_argument("Require finite S,K>0, finite r,q, and finite sigma,T>=0");
        }
    }
}

} // namespace

CallResult evaluate_call(const CallInputs& input, BatchBackend backend) {
    // Validate before entering par_unseq: an exception from a parallel element
    // callback would invoke std::terminate under standard execution policies.
    validate(input);
    CallResult result;
    const auto n = input.S.size();
    result.price.resize(n);
    result.delta.resize(n);
    result.gamma.resize(n);

    if (backend == BatchBackend::Serial) {
        for (std::size_t i = 0; i < n; ++i) {
            result.price[i] = bs_price_call(input.S[i], input.K[i], input.r[i], input.q[i], input.sigma[i], input.T[i]);
            result.delta[i] = delta_call(input.S[i], input.K[i], input.r[i], input.q[i], input.sigma[i], input.T[i]);
            result.gamma[i] = Gamma(input.S[i], input.K[i], input.r[i], input.q[i], input.sigma[i], input.T[i]);
        }
    } else if (backend == BatchBackend::ParUnseq) {
        bs_price_call_par(input.S, input.K, input.r, input.q, input.sigma, input.T, result.price);
        delta_call_par(input.S, input.K, input.r, input.q, input.sigma, input.T, result.delta);
        Gamma_par(input.S, input.K, input.r, input.q, input.sigma, input.T, result.gamma);
    } else {
        throw std::invalid_argument("Unknown batch backend");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (input.sigma[i] == 0.0 || input.T[i] == 0.0) {
            const double a = input.S[i] * std::exp(-input.q[i] * input.T[i]);
            const double b = input.K[i] * std::exp(-input.r[i] * input.T[i]);
            result.price[i] = std::max(a - b, 0.0);
            result.delta[i] = std::exp(-input.q[i] * input.T[i]) *
                (a > b ? 1.0 : a == b ? 0.5 : 0.0);
            result.gamma[i] = 0.0;
        }
        if (!std::isfinite(result.price[i]) || !std::isfinite(result.delta[i]) ||
            !std::isfinite(result.gamma[i])) {
            throw std::overflow_error("Black-Scholes result is not representable as double");
        }
    }
    return result;
}

CallResult evaluate_call_complex_step(const CallInputs& input, const std::vector<double>& h, int order) {
    if (order < 2 || order > 6) throw std::invalid_argument("order must be 2, 3, 4, 5, 6");
    if (h.size() != input.S.size()) throw std::invalid_argument("Step array must match batch length");
    for (double step : h) {
        if (!std::isfinite(step) || step <= 0) throw std::invalid_argument("Steps must be positive and finite");
    }
    auto result = evaluate_call(input);
    for (std::size_t i = 0; i < input.S.size(); ++i) {
        if (input.sigma[i] == 0 || input.T[i] == 0) continue;
        result.delta[i] = delta_complex_step(input.S[i], input.K[i], input.r[i], input.q[i], input.sigma[i], input.T[i], h[i], order);
        result.gamma[i] = gamma_45_deg(input.S[i], input.K[i], input.r[i], input.q[i], input.sigma[i], input.T[i], h[i], order);
    }
    return result;
}

CallResult evaluate_call_complex_step(const CallInputs& input, double h, int order) {
    if (!std::isfinite(h) || h <= 0) throw std::invalid_argument("h must be positive and finite");
    return evaluate_call_complex_step(input, std::vector<double>(input.S.size(), h), order);
}

} // namespace black_scholes
