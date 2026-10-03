// End-to-end throughput of the current serial and par_unseq call facades.
// No pass/fail speed threshold: this is a measurement tool, not a unit test.
#include "bs_batch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace bs = black_scholes;

namespace {

volatile double observed = 0.0;

double measure(const bs::CallInputs& input, bs::BatchBackend backend, int repeats) {
    using clock = std::chrono::steady_clock;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    const auto warm = bs::evaluate_call(input, backend);
    observed += warm.price[warm.price.size() / 2];
    for (int i = 0; i < repeats; ++i) {
        const auto start = clock::now();
        const auto output = bs::evaluate_call(input, backend);
        const auto end = clock::now();
        observed += output.price[output.price.size() / 2];
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 7;
    if (repeats < 3 || repeats % 2 == 0) {
        std::cerr << "Usage: bs_batch_benchmark [odd_repeats>=3]\n";
        return 2;
    }
    std::cout << "backend,n,median_ms,options_per_s,speedup_vs_serial\n";
    for (const std::size_t n : {256u, 16384u, 262144u}) {
        std::vector<double> S(n), K(n), r(n), q(n), sigma(n), T(n);
        for (std::size_t i = 0; i < n; ++i) {
            S[i] = 75.0 + static_cast<double>(i % 57);
            K[i] = 80.0 + static_cast<double>(i % 41);
            r[i] = -0.01 + 0.001 * static_cast<double>(i % 50);
            q[i] = 0.001 * static_cast<double>(i % 20);
            sigma[i] = 0.05 + 0.01 * static_cast<double>(i % 40);
            T[i] = 1.0 / 365.0 + 0.01 * static_cast<double>(i % 100);
        }
        const bs::CallInputs input{S, K, r, q, sigma, T};
        const auto a = bs::evaluate_call(input, bs::BatchBackend::Serial);
        const auto b = bs::evaluate_call(input, bs::BatchBackend::ParUnseq);
        for (std::size_t i = 0; i < n; ++i) {
            if (std::abs(a.price[i] - b.price[i]) > 2e-11 ||
                std::abs(a.delta[i] - b.delta[i]) > 2e-11 ||
                std::abs(a.gamma[i] - b.gamma[i]) > 2e-11) {
                std::cerr << "Backend mismatch before benchmark\n";
                return 1;
            }
        }
        const double serial_ms = measure(input, bs::BatchBackend::Serial, repeats);
        const double parallel_ms = measure(input, bs::BatchBackend::ParUnseq, repeats);
        std::cout << std::fixed << std::setprecision(6)
                  << "serial," << n << ',' << serial_ms << ',' << n * 1000.0 / serial_ms << ",1.000\n"
                  << "par_unseq," << n << ',' << parallel_ms << ',' << n * 1000.0 / parallel_ms
                  << ',' << serial_ms / parallel_ms << '\n';
    }
    if (!std::isfinite(observed)) return 1;
    return 0;
}
