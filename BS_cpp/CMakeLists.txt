cmake_minimum_required(VERSION 3.15)
project(quant_bs_greeks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(black_scholes_core STATIC
    black_scholes.cpp
    bs_batch.cpp
    complex_step.cpp
)
target_include_directories(black_scholes_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(Phi_real_test
    tests/Phi_real_test.cpp
)
target_link_libraries(Phi_real_test PRIVATE black_scholes_core)

add_executable(gamma_edge_test
    tests/gamma_edge_test.cpp
)
target_link_libraries(gamma_edge_test PRIVATE black_scholes_core)

add_executable(bs_greeks_test
    experiments/assignment_sweep.cpp
)
target_link_libraries(bs_greeks_test PRIVATE black_scholes_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(complex_cdf_order_test tests/complex_cdf_order_test.cpp)
    target_link_libraries(complex_cdf_order_test PRIVATE black_scholes_core)
    add_test(NAME complex_cdf_order
        COMMAND complex_cdf_order_test "${CMAKE_CURRENT_SOURCE_DIR}/../testdata/complex_cdf_golden.csv")
    add_executable(bs_batch_contract_test tests/bs_batch_contract_test.cpp)
    target_link_libraries(bs_batch_contract_test PRIVATE black_scholes_core)
    add_test(NAME bs_batch_contract
        COMMAND bs_batch_contract_test "${CMAKE_CURRENT_SOURCE_DIR}/../testdata/bs_call_golden.csv")
    add_test(NAME gamma_edge COMMAND gamma_edge_test)
    add_test(NAME normal_cdf COMMAND Phi_real_test)
endif()

option(BS_BUILD_BENCHMARKS "Build manual throughput benchmarks" OFF)
if(BS_BUILD_BENCHMARKS)
    add_executable(bs_batch_benchmark benchmarks/bs_batch_benchmark.cpp)
    target_link_libraries(bs_batch_benchmark PRIVATE black_scholes_core)
endif()

