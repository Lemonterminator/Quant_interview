cmake -S BS_cpp -B BS_cpp/build
cmake --build BS_cpp/build --config Release
BS_cpp/build/Release/bs_greeks_test.exe
python Python_prototype/plot_cpp_results.py
