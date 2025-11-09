cmake_minimum_required(VERSION 3.15)
project(quant_bs_greeks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(black_scholes_core STATIC
    black_scholes.cpp
)
target_include_directories(black_scholes_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(Phi_real_test
    tests/Phi_real_test.cpp
)
target_link_libraries(Phi_real_test PRIVATE black_scholes_core)

add_executable(bs_greeks_test
    tests/black_scholes_test.cpp
)
target_link_libraries(bs_greeks_test PRIVATE black_scholes_core)

