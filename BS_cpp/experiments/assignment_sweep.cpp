#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <cctype>
#include <string>
#include <vector>

#include "black_scholes.h"
#include "complex_step.h"

namespace {

using black_scholes::Gamma;
using black_scholes::bs_price_call;
using black_scholes::delta_call;
using black_scholes::delta_forward;
using black_scholes::gamma_forward;
using black_scholes::delta_complex_step;
using black_scholes::gamma_cs_real;
using black_scholes::gamma_45_deg;

constexpr double HREL_MIN = 1e-16;
constexpr double HREL_MAX = 1e-4;
constexpr std::size_t HREL_POINTS = 24;
const std::filesystem::path RESULTS_ROOT{"artifacts/assignment/cpp"};
constexpr double FLOAT_EPS = std::numeric_limits<double>::epsilon();

struct Scenario {
    std::string name;
    double S;
    double K;
    double r;
    double q;
    double sigma;
    double T;
};

struct Summary {
    double max;
    double median;
    double p99;
};

struct DeltaSweep {
    std::vector<double> hrel;
    std::vector<double> h;
    std::vector<double> delta_ref;
    std::vector<double> delta_fd;
    std::vector<double> delta_cs;
    std::vector<double> delta_fd_abs_err;
    std::vector<double> delta_cs_abs_err;
    std::vector<double> delta_fd_rel_err;
    std::vector<double> delta_cs_rel_err;
};

struct GammaSweep {
    std::vector<double> hrel;
    std::vector<double> h;
    std::vector<double> gamma_ref;
    std::vector<double> gamma_fd;
    std::vector<double> gamma_cs_real;
    std::vector<double> gamma_45_deg;
    std::vector<double> gamma_fd_abs_err;
    std::vector<double> gamma_cs_abs_err;
    std::vector<double> gamma_45_abs_err;
    std::vector<double> gamma_fd_rel_err;
    std::vector<double> gamma_cs_rel_err;
    std::vector<double> gamma_45_rel_err;
};

struct StressDataset {
    std::vector<double> S;
    std::vector<double> K;
    std::vector<double> r;
    std::vector<double> q;
    std::vector<double> sigma;
    std::vector<double> T;
};

struct RoutingInfo {
    std::vector<double> h_rel;
    std::vector<double> tau;
    std::vector<double> moneyness;
    std::vector<char> ultra_short_tau;
    std::vector<char> short_tau;
    std::vector<char> large_tau;
    std::vector<char> deep_otm_itm;
    std::vector<char> extreme_otm_itm;
    std::vector<char> atm_tiny_sigma;
    double atm_share{};
};

std::vector<double> logspace(double start_exp, double end_exp, std::size_t points) {
    std::vector<double> out(points);
    const double step = (end_exp - start_exp) / static_cast<double>(points - 1);
    for (std::size_t i = 0; i < points; ++i) {
        out[i] = std::pow(10.0, start_exp + step * static_cast<double>(i));
    }
    return out;
}

std::string slugify(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        if (std::isalnum(c)) { return static_cast<char>(std::tolower(c)); }
        return '_';
    });
    text.erase(std::unique(text.begin(), text.end(), [](char a, char b){ return a == '_' && b == '_'; }), text.end());
    if (!text.empty() && text.front() == '_') text.erase(text.begin());
    if (!text.empty() && text.back() == '_') text.pop_back();
    return text;
}

double relative_error(double value, double reference) {
    const double denom = std::max(std::abs(reference), FLOAT_EPS);
    return std::abs(value - reference) / denom;
}

double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = pct * (static_cast<double>(values.size()) - 1.0);
    const std::size_t idx = static_cast<std::size_t>(rank);
    const double frac = rank - static_cast<double>(idx);
    if (idx + 1 < values.size()) {
        return values[idx] + frac * (values[idx + 1] - values[idx]);
    }
    return values[idx];
}

Summary summarize(const std::vector<double>& values) {
    Summary stats{};
    if (values.empty()) {
        return stats;
    }
    stats.max = *std::max_element(values.begin(), values.end());
    stats.median = percentile(values, 0.5);
    stats.p99 = percentile(values, 0.99);
    return stats;
}

DeltaSweep run_delta_sweep(const Scenario& scenario, const std::vector<double>& hrel_grid, int order = 3) {
    DeltaSweep sweep;
    sweep.hrel = hrel_grid;
    sweep.h.resize(hrel_grid.size());
    sweep.delta_ref.assign(hrel_grid.size(), delta_call(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T));
    sweep.delta_fd.resize(hrel_grid.size());
    sweep.delta_cs.resize(hrel_grid.size());
    sweep.delta_fd_abs_err.resize(hrel_grid.size());
    sweep.delta_cs_abs_err.resize(hrel_grid.size());
    sweep.delta_fd_rel_err.resize(hrel_grid.size());
    sweep.delta_cs_rel_err.resize(hrel_grid.size());

    for (std::size_t i = 0; i < hrel_grid.size(); ++i) {
        const double h_abs = hrel_grid[i] * scenario.S;
        sweep.h[i] = h_abs;
        sweep.delta_fd[i] = delta_forward(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h_abs);
        sweep.delta_cs[i] = delta_complex_step(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h_abs, order);
        sweep.delta_fd_abs_err[i] = std::abs(sweep.delta_fd[i] - sweep.delta_ref[i]);
        sweep.delta_cs_abs_err[i] = std::abs(sweep.delta_cs[i] - sweep.delta_ref[i]);
        sweep.delta_fd_rel_err[i] = relative_error(sweep.delta_fd[i], sweep.delta_ref[i]);
        sweep.delta_cs_rel_err[i] = relative_error(sweep.delta_cs[i], sweep.delta_ref[i]);
    }
    return sweep;
}

GammaSweep run_gamma_sweep(const Scenario& scenario, const std::vector<double>& hrel_grid, int order = 3) {
    GammaSweep sweep;
    sweep.hrel = hrel_grid;
    sweep.h.resize(hrel_grid.size());
    sweep.gamma_ref.assign(hrel_grid.size(), Gamma(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T));
    sweep.gamma_fd.resize(hrel_grid.size());
    sweep.gamma_cs_real.resize(hrel_grid.size());
    sweep.gamma_45_deg.resize(hrel_grid.size());
    sweep.gamma_fd_abs_err.resize(hrel_grid.size());
    sweep.gamma_cs_abs_err.resize(hrel_grid.size());
    sweep.gamma_45_abs_err.resize(hrel_grid.size());
    sweep.gamma_fd_rel_err.resize(hrel_grid.size());
    sweep.gamma_cs_rel_err.resize(hrel_grid.size());
    sweep.gamma_45_rel_err.resize(hrel_grid.size());

    for (std::size_t i = 0; i < hrel_grid.size(); ++i) {
        const double h_abs = hrel_grid[i] * scenario.S;
        sweep.h[i] = h_abs;
        sweep.gamma_fd[i] = gamma_forward(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h_abs);
        sweep.gamma_cs_real[i] = gamma_cs_real(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h_abs, order);
        sweep.gamma_45_deg[i] = gamma_45_deg(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h_abs, order);

        sweep.gamma_fd_abs_err[i] = std::abs(sweep.gamma_fd[i] - sweep.gamma_ref[i]);
        sweep.gamma_cs_abs_err[i] = std::abs(sweep.gamma_cs_real[i] - sweep.gamma_ref[i]);
        sweep.gamma_45_abs_err[i] = std::abs(sweep.gamma_45_deg[i] - sweep.gamma_ref[i]);
        sweep.gamma_fd_rel_err[i] = relative_error(sweep.gamma_fd[i], sweep.gamma_ref[i]);
        sweep.gamma_cs_rel_err[i] = relative_error(sweep.gamma_cs_real[i], sweep.gamma_ref[i]);
        sweep.gamma_45_rel_err[i] = relative_error(sweep.gamma_45_deg[i], sweep.gamma_ref[i]);
    }
    return sweep;
}

void ensure_directory(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

void write_greeks_csv(const std::filesystem::path& path, const DeltaSweep& delta, const GammaSweep& gamma) {
    ensure_directory(path.parent_path());
    std::ofstream out(path);
    out << "h_rel,h,Delta_analytic,Delta_fd,Delta_cs,err_D_fd,err_D_cs,Gamma_analytic,Gamma_fd,Gamma_cs_real,Gamma_cs_45,err_G_fd,err_G_cs_real,err_G_cs_45\n";
    out << std::scientific << std::setprecision(12);
    const std::size_t rows = delta.hrel.size();
    for (std::size_t i = 0; i < rows; ++i) {
        out << delta.hrel[i] << ','
            << delta.h[i] << ','
            << delta.delta_ref[i] << ','
            << delta.delta_fd[i] << ','
            << delta.delta_cs[i] << ','
            << delta.delta_fd_abs_err[i] << ','
            << delta.delta_cs_abs_err[i] << ','
            << gamma.gamma_ref[i] << ','
            << gamma.gamma_fd[i] << ','
            << gamma.gamma_cs_real[i] << ','
            << gamma.gamma_45_deg[i] << ','
            << gamma.gamma_fd_abs_err[i] << ','
            << gamma.gamma_cs_abs_err[i] << ','
            << gamma.gamma_45_abs_err[i] << '\n';
    }
}

void print_error_summary(const std::string& title, const std::vector<std::pair<std::string, std::vector<double>>> &series) {
    std::cout << title << '\n';
    std::cout << "method              max_abs_err     median_abs_err      p99_abs_err" << '\n';
    std::cout << "-------------------------------------------------------------------" << '\n';
    std::cout << std::scientific << std::setprecision(3);
    for (const auto& entry : series) {
        Summary stats = summarize(entry.second);
        std::cout << std::setw(18) << std::left << entry.first
                  << std::setw(14) << std::right << stats.max
                  << std::setw(18) << stats.median
                  << std::setw(18) << stats.p99 << '\n';
    }
    std::cout.unsetf(std::ios::floatfield);
}

std::vector<double> geomspace(double start, double end, std::size_t points) {
    std::vector<double> out(points);
    const double log_start = std::log(start);
    const double log_end = std::log(end);
    const double step = (log_end - log_start) / static_cast<double>(points - 1);
    for (std::size_t i = 0; i < points; ++i) {
        out[i] = std::exp(log_start + step * static_cast<double>(i));
    }
    return out;
}

template <typename Dist>
std::vector<double> sample(std::mt19937_64& rng, Dist dist, std::size_t size) {
    std::vector<double> out(size);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

StressDataset generate_stress_test_dataset(std::size_t n, std::uint64_t seed = 1234) {
    std::mt19937_64 rng(seed);

    const std::size_t m = n / 4;
    const std::size_t k = n / 4;
    const std::size_t h = n / 6;
    const std::size_t p = 128;
    const std::size_t total = n + m + k + h + p;

    StressDataset data;
    data.S.reserve(total);
    data.K.reserve(total);
    data.r.reserve(total);
    data.q.reserve(total);
    data.sigma.reserve(total);
    data.T.reserve(total);

    auto append_block = [&](const std::vector<double>& S_block,
                            const std::vector<double>& K_block,
                            const std::vector<double>& r_block,
                            const std::vector<double>& q_block,
                            const std::vector<double>& sigma_block,
                            const std::vector<double>& T_block) {
        data.S.insert(data.S.end(), S_block.begin(), S_block.end());
        data.K.insert(data.K.end(), K_block.begin(), K_block.end());
        data.r.insert(data.r.end(), r_block.begin(), r_block.end());
        data.q.insert(data.q.end(), q_block.begin(), q_block.end());
        data.sigma.insert(data.sigma.end(), sigma_block.begin(), sigma_block.end());
        data.T.insert(data.T.end(), T_block.begin(), T_block.end());
    };

    auto S_reg = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.25), n);
    auto K_reg = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.25), n);
    auto r_reg = sample(rng, std::uniform_real_distribution<double>(-0.01, 0.05), n);
    auto q_reg = sample(rng, std::uniform_real_distribution<double>(0.0, 0.03), n);
    auto sig_reg = sample(rng, std::uniform_real_distribution<double>(0.05, 0.60), n);
    auto T_reg = sample(rng, std::uniform_real_distribution<double>(1.0 / 365.0, 2.0), n);
    append_block(S_reg, K_reg, r_reg, q_reg, sig_reg, T_reg);

    auto S_ne = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.05), m);
    std::vector<double> K_ne(m);
    {
        auto bumps = sample(rng, std::lognormal_distribution<double>(0.0, 0.001), m);
        for (std::size_t i = 0; i < m; ++i) { K_ne[i] = S_ne[i] * bumps[i]; }
    }
    auto T_ne = sample(rng, std::uniform_real_distribution<double>(1.0 / 3650.0, 3.0 / 365.0), m);
    auto sig_ne = sample(rng, std::uniform_real_distribution<double>(1e-3, 8e-3), m);
    auto r_ne = sample(rng, std::uniform_real_distribution<double>(-0.01, 0.03), m);
    auto q_ne = sample(rng, std::uniform_real_distribution<double>(0.0, 0.02), m);
    append_block(S_ne, K_ne, r_ne, q_ne, sig_ne, T_ne);

    auto S_deep = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.25), k);
    std::vector<double> K_deep(k);
    {
        std::uniform_real_distribution<double> mag_dist(1.0, 2.5);
        std::bernoulli_distribution sign_dist(0.5);
        for (std::size_t i = 0; i < k; ++i) {
            const double mag = mag_dist(rng);
            const double sign = sign_dist(rng) ? 1.0 : -1.0;
            const double lm = sign * mag;
            K_deep[i] = S_deep[i] * std::exp(-lm);
        }
    }
    auto T_deep = sample(rng, std::uniform_real_distribution<double>(5.0 / 365.0, 1.5), k);
    auto sig_deep = sample(rng, std::uniform_real_distribution<double>(0.10, 0.60), k);
    auto r_deep = sample(rng, std::uniform_real_distribution<double>(-0.01, 0.05), k);
    auto q_deep = sample(rng, std::uniform_real_distribution<double>(0.0, 0.03), k);
    append_block(S_deep, K_deep, r_deep, q_deep, sig_deep, T_deep);

    auto S_hv = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.25), h);
    auto K_hv = sample(rng, std::lognormal_distribution<double>(std::log(100.0), 0.25), h);
    auto T_hv = sample(rng, std::uniform_real_distribution<double>(1.0, 5.0), h);
    auto sig_hv = sample(rng, std::uniform_real_distribution<double>(0.6, 1.2), h);
    auto r_hv = sample(rng, std::uniform_real_distribution<double>(-0.005, 0.04), h);
    auto q_hv = sample(rng, std::uniform_real_distribution<double>(0.0, 0.03), h);
    append_block(S_hv, K_hv, r_hv, q_hv, sig_hv, T_hv);

    const auto T_pt = geomspace(1e-6, 1e-3, p);
    const auto sig_pt = geomspace(1e-4, 1e-2, p);
    std::vector<double> S_pt(p, 100.0);
    std::vector<double> K_pt(p, 100.0);
    std::vector<double> r_pt(p, 0.0);
    std::vector<double> q_pt(p, 0.0);
    append_block(S_pt, K_pt, r_pt, q_pt, sig_pt, T_pt);

    return data;
}

RoutingInfo route_hrel(const std::vector<double>& S,
                       const std::vector<double>& K,
                       const std::vector<double>& sigma,
                       const std::vector<double>& T) {
    const std::size_t n = S.size();
    RoutingInfo info;
    info.h_rel.assign(n, 1e-8);
    info.tau.resize(n);
    info.moneyness.resize(n);
    info.ultra_short_tau.assign(n, 0);
    info.short_tau.assign(n, 0);
    info.large_tau.assign(n, 0);
    info.deep_otm_itm.assign(n, 0);
    info.extreme_otm_itm.assign(n, 0);
    info.atm_tiny_sigma.assign(n, 0);

    std::size_t atm_count = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const double tau = sigma[i] * std::sqrt(std::max(T[i], 0.0));
        const double mny = std::abs(std::log(std::max(S[i], 1e-12) / std::max(K[i], 1e-12)));
        info.tau[i] = tau;
        info.moneyness[i] = mny;
        const bool atm = mny <= 1e-3;
        if (atm) { ++atm_count; }

        if (tau < 1e-4) {
            info.ultra_short_tau[i] = 1;
            info.h_rel[i] = 5e-12;
        } else if (tau < 1e-3) {
            info.short_tau[i] = 1;
            info.h_rel[i] = 5e-11;
        }

        if (tau > 0.5) {
            info.large_tau[i] = 1;
            info.h_rel[i] = std::max(info.h_rel[i], 2e-8);
        }
        if (mny > 0.5) {
            info.deep_otm_itm[i] = 1;
            info.h_rel[i] = std::max(info.h_rel[i], 5e-8);
        }
        if (mny > 1.0) {
            info.extreme_otm_itm[i] = 1;
            info.h_rel[i] = std::max(info.h_rel[i], 2e-7);
        }
        if (atm && sigma[i] < 2e-3) {
            info.atm_tiny_sigma[i] = 1;
            info.h_rel[i] = std::min(info.h_rel[i], 1e-12);
        }
    }

    for (double& value : info.h_rel) {
        value = std::clamp(value, HREL_MIN, 5e-5);
    }

    info.atm_share = static_cast<double>(atm_count) / static_cast<double>(n);
    return info;
}

double median_for_mask(const std::vector<double>& data, const std::vector<char>& mask) {
    std::vector<double> subset;
    subset.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (mask[i]) {
            subset.push_back(data[i]);
        }
    }
    if (subset.empty()) {
        return 0.0;
    }
    return percentile(std::move(subset), 0.5);
}

void print_router_diagnostics(const RoutingInfo& info) {
    const std::size_t n = info.h_rel.size();
    auto quantiles = [&](const std::vector<double>& values) {
        return std::array<double, 5>{
            percentile(values, 0.0),
            percentile(values, 0.25),
            percentile(values, 0.5),
            percentile(values, 0.75),
            percentile(values, 1.0)
        };
    };
    const auto stats = quantiles(info.h_rel);
    const auto tau_min = *std::min_element(info.tau.begin(), info.tau.end());
    const auto tau_max = *std::max_element(info.tau.begin(), info.tau.end());
    const auto mon_min = *std::min_element(info.moneyness.begin(), info.moneyness.end());
    const auto mon_max = *std::max_element(info.moneyness.begin(), info.moneyness.end());

    std::cout << "\nRouter h_rel stats: min=" << stats[0]
              << ", q25=" << stats[1]
              << ", median=" << stats[2]
              << ", q75=" << stats[3]
              << ", max=" << stats[4] << '\n';
    std::cout << "tau range [" << tau_min << ", " << tau_max
              << "], moneyness range [" << mon_min << ", " << mon_max
              << "], ATM share " << info.atm_share * 100.0 << "%" << '\n';

    auto report_segment = [&](const std::string& name, const std::vector<char>& mask) {
        const std::size_t count = static_cast<std::size_t>(std::count(mask.begin(), mask.end(), 1));
        if (count == 0) { return; }
        const double pct = static_cast<double>(count) / static_cast<double>(n);
        std::cout << "  segment " << std::setw(18) << std::left << name
                  << std::setw(0) << std::right << " count=" << std::setw(6) << count
                  << " (" << std::fixed << std::setprecision(2) << pct * 100.0 << "%), h_rel~"
                  << std::scientific << median_for_mask(info.h_rel, mask) << std::defaultfloat << '\n';
    };

    report_segment("ultra_short_tau", info.ultra_short_tau);
    report_segment("short_tau", info.short_tau);
    report_segment("large_tau", info.large_tau);
    report_segment("deep_otm_itm", info.deep_otm_itm);
    report_segment("extreme_otm_itm", info.extreme_otm_itm);
    report_segment("atm_tiny_sigma", info.atm_tiny_sigma);
}

std::vector<double> delta_cs_batch(const StressDataset& data, const std::vector<double>& h_abs, int order = 3) {
    std::vector<double> out(data.S.size());
    for (std::size_t i = 0; i < data.S.size(); ++i) {
        out[i] = delta_complex_step(data.S[i], data.K[i], data.r[i], data.q[i], data.sigma[i], data.T[i], h_abs[i], order);
    }
    return out;
}

std::vector<double> gamma_45_batch(const StressDataset& data, const std::vector<double>& h_abs, int order = 3) {
    std::vector<double> out(data.S.size());
    for (std::size_t i = 0; i < data.S.size(); ++i) {
        out[i] = gamma_45_deg(data.S[i], data.K[i], data.r[i], data.q[i], data.sigma[i], data.T[i], h_abs[i], order);
    }
    return out;
}

void run_parallel_stress_test(std::size_t batch_size = 40000, int order = 3) {
    const auto dataset = generate_stress_test_dataset(batch_size);
    const RoutingInfo routing = route_hrel(dataset.S, dataset.K, dataset.sigma, dataset.T);

    std::vector<double> h_abs(dataset.S.size());
    for (std::size_t i = 0; i < dataset.S.size(); ++i) {
        const double S_safe = std::max(dataset.S[i], 1e-12);
        h_abs[i] = routing.h_rel[i] * S_safe;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto delta = delta_cs_batch(dataset, h_abs, order);
    const auto gamma = gamma_45_batch(dataset, h_abs, order);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\n----------------------------------------------------\n";
    std::cout << "Stress batch size " << dataset.S.size() << " (requested " << batch_size << ") evaluated in " << std::fixed << std::setprecision(3)
              << elapsed << " seconds." << '\n';
    std::cout << "----------------------------------------------------\n";
    print_router_diagnostics(routing);

    auto summarize_vec = [](const std::vector<double>& v) {
        const auto minmax = std::minmax_element(v.begin(), v.end());
        const double med = percentile(v, 0.5);
        return std::tuple<double, double, double>(*minmax.first, med, *minmax.second);
    };
    const auto [d_min, d_med, d_max] = summarize_vec(delta);
    const auto [g_min, g_med, g_max] = summarize_vec(gamma);

    std::cout << std::scientific << std::setprecision(3)
              << "delta stats -> min " << d_min << ", median " << d_med << ", max " << d_max << '\n'
              << "gamma stats -> min " << g_min << ", median " << g_med << ", max " << g_max << '\n';
    std::cout.unsetf(std::ios::floatfield);
}

} // namespace

int main(int argc, char** argv) {
    int order = 3;
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: bs_greeks_test [--order 2|3|4|5|6]\n";
        return 0;
    }
    if (argc != 1) {
        try {
            if (argc != 3 || std::string(argv[1]) != "--order") throw std::invalid_argument("arguments");
            std::size_t consumed = 0;
            order = std::stoi(argv[2], &consumed);
            if (consumed != std::string(argv[2]).size() || order < 2 || order > 6) throw std::invalid_argument("order");
        } catch (const std::exception&) {
            std::cerr << "Usage: bs_greeks_test [--order 2|3|4|5|6]\n";
            return 2;
        }
    }
    std::cout << "CDF Taylor order: P" << order << '\n';
    const std::vector<Scenario> scenarios = {
        {"ATM reference S=100 K=100 r=0 q=0 sigma=20% T=1y", 100.0, 100.0, 0.0, 0.0, 0.20, 1.0},
        {"Near-expiry low-vol ATM S=100 K=100 r=0 q=0 sigma=1% T=1/365", 100.0, 100.0, 0.0, 0.0, 0.01, 1.0 / 365.0}
    };

    const auto hrel_grid = logspace(std::log10(HREL_MIN), std::log10(HREL_MAX), HREL_POINTS);

    for (const auto& scenario : scenarios) {
        const std::string slug = slugify(scenario.name);
        const std::filesystem::path scenario_dir = RESULTS_ROOT / ("p" + std::to_string(order)) / slug;
        ensure_directory(scenario_dir);

        std::cout << "\nScenario: " << scenario.name << '\n';

        const auto delta_sweep = run_delta_sweep(scenario, hrel_grid, order);
        print_error_summary("\nDelta absolute error summary",
                            {
                                {"delta forward", delta_sweep.delta_fd_abs_err},
                                {"delta complex-step", delta_sweep.delta_cs_abs_err}
                            });

        const auto gamma_sweep = run_gamma_sweep(scenario, hrel_grid, order);
        print_error_summary("\nGamma absolute error summary",
                            {
                                {"gamma forward", gamma_sweep.gamma_fd_abs_err},
                                {"gamma cs real", gamma_sweep.gamma_cs_abs_err},
                                {"gamma 45deg", gamma_sweep.gamma_45_abs_err}
                            });

        const auto combined_csv = scenario_dir / "bs_fd_vs_complex.csv";
        write_greeks_csv(combined_csv, delta_sweep, gamma_sweep);
        std::cout << "\nSaved combined CSV to " << std::filesystem::absolute(combined_csv) << '\n';
    }

    run_parallel_stress_test(40'000, order);
    return 0;
}
