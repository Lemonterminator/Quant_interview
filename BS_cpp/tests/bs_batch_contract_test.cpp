#include "bs_batch.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bs = black_scholes;

namespace {

struct Dataset {
    std::vector<std::string> names;
    std::vector<double> S, K, r, q, sigma, T, price, delta, gamma;
};

Dataset read_golden(const char* path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open golden CSV");
    std::string line;
    std::getline(stream, line); // fixed, documented header
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "case,S,K,r,q,sigma,T,price,delta,gamma") {
        throw std::runtime_error("Unexpected golden CSV header");
    }
    Dataset d;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(row, field, ',')) fields.push_back(field);
        if (fields.size() != 10) throw std::runtime_error("Malformed golden CSV row");
        d.names.push_back(fields[0]);
        d.S.push_back(std::stod(fields[1]));
        d.K.push_back(std::stod(fields[2]));
        d.r.push_back(std::stod(fields[3]));
        d.q.push_back(std::stod(fields[4]));
        d.sigma.push_back(std::stod(fields[5]));
        d.T.push_back(std::stod(fields[6]));
        d.price.push_back(std::stod(fields[7]));
        d.delta.push_back(std::stod(fields[8]));
        d.gamma.push_back(std::stod(fields[9]));
    }
    if (d.names.size() < 8) throw std::runtime_error("Golden CSV is incomplete");
    return d;
}

bool close(double actual, double expected) {
    const double tolerance = std::abs(expected) < 1e-4
        ? 1e-15 + 1e-8 * std::abs(expected)
        : 5e-13 + 5e-11 * std::abs(expected);
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

int check(const std::vector<double>& actual, const std::vector<double>& expected,
          const std::vector<std::string>& names, const char* backend, const char* field) {
    int failures = 0;
    if (actual.size() != expected.size()) return 1;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!close(actual[i], expected[i])) {
            std::cerr << backend << " " << field << " " << names[i] << ": got "
                      << actual[i] << ", expected " << expected[i] << '\n';
            ++failures;
        }
    }
    return failures;
}

template <typename F>
bool rejects_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: bs_batch_contract_test GOLDEN_CSV\n";
        return 2;
    }
    try {
        const Dataset d = read_golden(argv[1]);
        const bs::CallInputs input{d.S, d.K, d.r, d.q, d.sigma, d.T};
        int failures = 0;
        for (const auto backend : {bs::BatchBackend::Serial, bs::BatchBackend::ParUnseq}) {
            const char* label = backend == bs::BatchBackend::Serial ? "serial" : "par_unseq";
            const auto output = bs::evaluate_call(input, backend);
            failures += check(output.price, d.price, d.names, label, "price");
            failures += check(output.delta, d.delta, d.names, label, "delta");
            failures += check(output.gamma, d.gamma, d.names, label, "gamma");
        }

        // A non-power-of-two batch catches lane/tail and raw-pointer indexing mistakes.
        std::vector<double> S(1009), K(1009), r(1009), q(1009), sigma(1009), T(1009);
        for (std::size_t i = 0; i < S.size(); ++i) {
            S[i] = 75.0 + static_cast<double>(i % 57);
            K[i] = 80.0 + static_cast<double>(i % 41);
            r[i] = -0.01 + 0.001 * static_cast<double>(i % 50);
            q[i] = 0.001 * static_cast<double>(i % 20);
            sigma[i] = 0.05 + 0.01 * static_cast<double>(i % 40);
            T[i] = 1.0 / 365.0 + 0.01 * static_cast<double>(i % 100);
        }
        const bs::CallInputs varied{S, K, r, q, sigma, T};
        const auto serial = bs::evaluate_call(varied, bs::BatchBackend::Serial);
        const auto parallel = bs::evaluate_call(varied, bs::BatchBackend::ParUnseq);
        for (std::size_t i = 0; i < S.size(); ++i) {
            if (!close(serial.price[i], parallel.price[i]) ||
                !close(serial.delta[i], parallel.delta[i]) ||
                !close(serial.gamma[i], parallel.gamma[i])) ++failures;
        }

        std::vector<double> empty, one{1.0}, nan{std::numeric_limits<double>::quiet_NaN()};
        if (!bs::evaluate_call({empty, empty, empty, empty, empty, empty}).price.empty()) ++failures;
        if (!rejects_invalid([&] { bs::evaluate_call({one, empty, one, one, one, one}); })) ++failures;
        if (!rejects_invalid([&] { bs::evaluate_call({nan, one, one, one, one, one}); })) ++failures;
        if (!rejects_invalid([&] { bs::evaluate_call({one, one, one, one, one, one}, static_cast<bs::BatchBackend>(99)); })) ++failures;

        if (failures) std::cerr << "bs_batch_contract_test: " << failures << " failures\n";
        else std::cout << "bs_batch_contract_test: all cases passed\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "bs_batch_contract_test: " << e.what() << '\n';
        return 2;
    }
}
