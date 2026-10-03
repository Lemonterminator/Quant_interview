#include "complex_step.h"
#include "bs_batch.h"
#include "black_scholes.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bs = black_scholes;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Expected invalid_argument");
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected golden CSV path");
        std::ifstream stream(argv[1]);
        require(static_cast<bool>(stream), "Cannot read golden CSV");
        std::string line;
        std::getline(stream, line);
        int count = 0;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            std::string field;
            double values[5];
            for (auto& value : values) {
                require(static_cast<bool>(std::getline(row, field, ',')), "Malformed golden row");
                value = std::stod(field);
            }
            const auto actual = bs::Phi_t({values[1], values[2]}, static_cast<int>(values[0]));
            require(std::abs(actual.real() - values[3]) < 2e-15, "CDF real golden mismatch");
            require(std::abs(actual.imag() - values[4]) < 2e-17, "CDF imaginary golden mismatch");
            ++count;
        }
        require(count == 20, "Incomplete golden CSV");
        require(bs::Phi_t({.3,.04}) == bs::Phi_t({.3,.04},3), "Default must be P3");
        const double g = bs::gamma_45_deg(100,100,0,0,.2,1,.2);
        require(g == bs::gamma_45_deg(100,100,0,0,.2,1,.2,3), "Gamma default propagation");
        require(std::abs(g-bs::Gamma(100,100,0,0,.2,1)) < 5e-12, "Default Gamma accuracy");
        require(std::abs(g-bs::gamma_45_deg(100,100,0,0,.2,1,.2,2)) > 1e-7, "Order must affect Gamma");
        require(std::abs(bs::delta_complex_step(100,100,0,0,.2,1,1e-40)-bs::delta_call(100,100,0,0,.2,1)) < 1e-12,
                "Tiny imaginary parts must not be discarded");
        const bs::CallInputs inputs{{90,100,110},{100,100,100},{0,0,0},{0,0,0},{.2,.2,0},{1,1,1}};
        const std::vector<double> steps{.1,.2,.3};
        for (int order = 2; order <= 6; ++order) {
            const auto result = bs::evaluate_call_complex_step(inputs, steps, order);
            for (int i = 0; i < 2; ++i) {
                require(result.gamma[i] == bs::gamma_45_deg(inputs.S[i],100,0,0,.2,1,steps[i],order), "Batch order forwarding");
            }
            require(result.gamma[2] == 0 && result.delta[2] == 1, "Degenerate conventions");
            const double x=.3, truth=-x*std::exp(-x*x/2)/std::sqrt(2*std::acos(-1.));
            const std::complex<double> w(1/std::sqrt(2.),1/std::sqrt(2.));
            auto error = [&](double h) { return std::abs((bs::Phi_t(x+w*h,order)+bs::Phi_t(x-w*h,order)).imag()/(h*h)-truth); };
            require(std::abs(error(.04)/error(.02)-(order==2?4:16)) < .1, "Local convergence order");
        }
        rejects([&] { bs::Phi_t({0,0},7); });
        rejects([&] { bs::bs_price_call_complex({100,.1},100,0,0,.2,1,1); });
        rejects([&] { bs::delta_complex_step(100,100,0,0,.2,1,.1,7); });
        rejects([&] { bs::gamma_cs_real(100,100,0,0,.2,1,.1,7); });
        rejects([&] { bs::gamma_45_deg(100,100,0,0,.2,1,.1,7); });
        rejects([&] { bs::evaluate_call_complex_step(inputs,.1,7); });
        rejects([&] { bs::evaluate_call_complex_step(inputs,std::vector<double>{.1},3); });
        rejects([&] { bs::evaluate_call_complex_step(inputs,0.,3); });
        std::cout << "complex_cdf_order_test: golden, default, propagation, convergence passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
