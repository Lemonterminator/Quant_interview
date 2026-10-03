# Black–Scholes 批量计算：测试与 SIMD 研发框架

本阶段保留原作业的 `black_scholes.py`、`black_scholes.cpp` 及其公开函数。新增一层严格的 call batch API，把 Python、NumPy、C++ scalar 和 `par_unseq` 放在同一输入域及同一组输出下比较。Put、完整 Greeks 与 IV 以后接入这层接口；当前代码只实现 call/Delta/Gamma。

## 文件结构

| 文件 | 用途 |
| --- | --- |
| `Python_prototype/bs_batch.py` | NumPy 广播、输入验证、退化语义、call/Delta/Gamma 批量入口 |
| `BS_cpp/bs_batch.h/.cpp` | 对应的 C++17  与 `Serial`/`ParUnseq` 后端 |
| `testdata/bs_call_golden.csv` | 两种语言共同读取的固定真值案例 |
| `testdata/generate_bs_call_golden.py` | 用可选的 mpmath 80 位精度重新生成真值；测试时不运行 |
| `tests/` | Python 契约、性质及数值方法测试 |
| `BS_cpp/tests/bs_batch_contract_test.cpp` | C++ golden、尾批长度、串行/并行一致性测试 |
| `benchmarks/benchmark_python.py` | Python loop、NumPy、显式线程分块的吞吐测量 |
| `BS_cpp/benchmarks/bs_batch_benchmark.cpp` | C++ scalar 与现有 `par_unseq` 的吞吐测量 |

## 契约和准确性

输入要求有限的 `S,K>0`、有限的 `r,q`、有限的 `sigma,T>=0`。Python 遵循 NumPy 广播规则，C++ 的六个 vector 必须等长。`sigma=0` 或 `T=0` 时返回折现远期内在价值、`Gamma=0`；内在价值折点的 Delta 采用 `0.5 exp(-qT)` 约定。这是 API 约定，不声称折点存在普通导数。非有限输出视为表示范围错误。

Golden 数据用独立的 mpmath 公式计算，覆盖普通 ATM、短期低波动、非零 `r/q`、负利率、低价格 OTM、高波动及退化边界。测试误差同时看绝对误差和参考值足够大时的相对误差；极小价格使用单独的相对精度要求，避免宽松的统一绝对容差掩盖错误。性质测试检查价格界、波动率单调性及 `(S,K) -> (aS,aK)` 的价格/Delta/Gamma 齐次关系。

有限差分和 complex-step 单独测试，不把“某方法始终最准确”设为断言。扫 `h` 时先标出 `S+h==S` 的不可分辨点；Gamma 同时记录绝对误差和在 `|Gamma_ref|` 足够大时的相对误差。当前 `Phi_t` 是截断的复数 CDF：45° 公式对精确解析延拓可有四阶截断误差，但不能把这项性质直接赋给现有近似 `Phi_t`。测试用 `f=Phi` 确认 P2 的二阶误差与 P3–P6 的局部四阶误差；复数路径默认已切换 P3。新增 `evaluate_call_complex_step` 暴露 `h` 和 `order`，详见 [API 说明](complex_step_api.md)。

下一阶段按此顺序增加测试：put-call parity 和 Greeks 恒等式；`Theta` 的日历时间符号；IV 无套利界、单调性及 `price -> IV -> price` 往返；低 Vega 时的价格残差和波动率误差分开报告。新增 C++ SIMD 后端时复用整套 golden 与性质测试。

## 正确性命令

在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v
cmake -S BS_cpp -B out/bs_gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build out/bs_gcc --parallel 4
ctest --test-dir out/bs_gcc --output-on-failure
```

`out/` 已在 `.gitignore` 中。Windows/MSVC 可改用单独 build 目录和 Visual Studio 开发环境。不要清除原来的 `BS_cpp/build`；它可能包含用户现有结果。

## 性能测量

正确性先于计时。两个 benchmark 都预先生成输入、预热、对照后端结果、测多个 batch 大小，并输出中位数吞吐。基准不设置固定“必须快于”的 CI 门槛；处理器、频率、散热、编译器、执行库和线程调度会改变结果。

```powershell
python benchmarks/benchmark_python.py --sizes 256 4096 32768 262144 --repeats 7 --workers 2 4
cmake -S BS_cpp -B out/bs_gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DBS_BUILD_BENCHMARKS=ON
cmake --build out/bs_gcc --parallel 4
./out/bs_gcc/bs_batch_benchmark.exe 7
```

Python 的 NumPy 路径是数组级原生循环；**NumPy 向量化不等于多核并行**。`numpy_threads` 明确把只读输入分块交给线程池，输出拼接及调度计入时间。报告 `speedup = T_numpy/T_threads` 和 `efficiency = speedup/workers`。部分 NumPy/SciPy 操作可能释放 GIL，收益要实测。小 batch 应预期线程开销占主导；还要观察内存峰值，避免大数组复制抵消收益。

C++ 当前 `ParUnseq` 只是给标准库的执行策略请求；它既不保证使用多个核，也不保证生成 SIMD 指令。当前 benchmark 只能报告相对于 `Serial` 的端到端加速比，不能在未知实际 worker 数时声称并行效率。下一步在确认热点与精度预算后，再增加独立的显式 SIMD 后端；需要编译器向量化报告与反汇编共同验证实际指令，特别留意 `erfc`、分支和临时数组是否妨碍向量化。MSVC 可用 `/Qvec-report:2`，GCC 可用 `-fopt-info-vec-optimized -fopt-info-vec-missed`。更改浮点优化选项时，重新运行 golden/性质测试并记录环境。

现在六个 C++ 输入 vector 是按字段连续存储的 SoA 布局。显式 SIMD 后端应维持只读输入、无输出别名、每个 lane 独立；先处理普通参数区，再用标量尾循环处理不足一个向量宽度的元素。`1009` 个元素的契约测试专门覆盖非整倍数尾部。需要分别测算计算密集区、极端 moneyness 分支和较小 batch 的调度成本；不要仅凭一次 40,000 点计时判断 SIMD 收益。

## 本机验证边界（2026-10-03）

最新结果见 [自动测试报告](test_status.md)。本次默认 P3 运行中，Python 和新增 C++ 阶数测试通过；C++ 阶数与 Gamma 边界测试在最新运行通过；批量契约在前一轮运行通过，但最新运行与 `normal_cdf` 一起被 Windows Application Control 拒绝启动（错误 4551），本次均不计为通过。

外部依据：[NumPy ufunc 与广播](https://numpy.org/doc/stable/reference/ufuncs.html)、[NumPy 线程安全](https://numpy.org/doc/stable/reference/thread_safety.html)、[C++ 执行策略草案](https://eel.is/c++draft/algorithms#execpol.parunseq)、[MSVC 向量化报告](https://learn.microsoft.com/en-us/cpp/build/reference/qvec-report-auto-vectorizer-reporting-level?view=msvc-170)、[GCC 优化报告](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html)。
