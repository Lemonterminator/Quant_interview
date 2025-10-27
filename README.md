C++ Setup and Phi_real Test

Build with CMake. The repo provides a test `Phi_real_test` that validates the standard normal CDF implementation in `bs_call_price.h`.

Quick start

- Generate build files:
  - Visual Studio (2022): `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`
  - Visual Studio (2019): `cmake -S . -B build -G "Visual Studio 16 2019" -A x64`
  - Ninja: `cmake -S . -B build -G Ninja`
  - MinGW Makefiles: `cmake -S . -B build -G "MinGW Makefiles"`

- Build:
  - Multi-config generators (Visual Studio): `cmake --build build --config Release`
  - Single-config generators (Ninja/MinGW): `cmake --build build`

- Run the test executable:
  - Visual Studio: `build/Release/Phi_real_test.exe`
  - Ninja/MinGW: `build/Phi_real_test`

Files

- `bs_call_price.h`: Header-only helpers for Phi_real, phi, and Black–Scholes call price.
- `tests/Phi_real_test.cpp`: Minimal test program for Phi_real.
- `CMakeLists.txt`: CMake build configuration creating `Phi_real_test`.

Notes

- Requires a C++17 compiler (MSVC, Clang, or GCC) and CMake 3.15+.
- If you don’t have Ninja installed, pick the generator matching your toolchain as shown above.

