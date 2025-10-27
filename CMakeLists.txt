cmake_minimum_required(VERSION 3.15)
project(quant_bs_greeks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Test for Phi_real
add_executable(Phi_real_test
    tests/Phi_real_test.cpp
)

target_include_directories(Phi_real_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

