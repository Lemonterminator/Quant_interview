# 给 ChatGPT 的任务说明:Black–Scholes 数值库重构计划 + 学习材料检索

> 使用方式:把本文件全文粘贴给 ChatGPT(建议开启联网/搜索)。本文件自包含,ChatGPT 无需访问我的仓库。
> 回答语言:中文;技术术语保留英文;所有来源链接用原文。

---

## 0. 你(ChatGPT)要做什么

我正在重构并扩展一个 Black–Scholes 期权定价的 C++ 数值库(Python 版本做对照)。下面第 2–6 节是项目现状、已修 bug、重构与扩展计划、涉及的数学与数值工程知识点。

**你的任务不是写代码**,而是:

1. **找到并整理学习材料**,帮我真正理解这个计划背后的数学与数值工程。要求见第 7 节。
2. **审查计划本身**:如果你发现第 3–5 节里有数学错误、符号约定错误、或数值上站不住的设计,请先指出来(第 8 节)。
3. 输出一份按周安排的学习路径,把材料映射到计划的每一步(第 7.4 节)。

**硬性要求(防幻觉):**

- 只给你能确认真实存在的资料。每条必须有可打开的 URL 或完整出版信息(作者、年份、出版方/期刊/卷期)。
- 不要编造章节号、页码、定理编号。不确定就写"章节需自行核对",并标注置信度(高/中/低)。
- 如果你无法联网验证某个链接,明确写"未验证"。
- 第 7.5 节有一份我自己的**种子清单**,是我凭记忆列的,**未经验证**。请逐条核实,错的删掉或纠正,并补充更好的。

---

## 1. 我的背景(请据此校准难度;不对的地方请直接问我)

以下是我的假设,不是事实陈述,请按"中等偏上"校准并在开头确认:

- 在准备 quant / risk quant 方向的面试,正在自学期权定价与数值方法。
- 会 Python(NumPy/SciPy)和 C++17(能读写模板、`std::vector`、`<execution>` 并行算法)。
- 微积分、线性代数、概率论(正态分布、条件期望、鞅的基本概念)没问题;随机微积分(Itô 引理)有接触,但不要假设我能熟练推导。
- 数值分析、浮点算术、舍入误差分析是我最想补强的部分。
- 偏好:免费或开放获取 > 教材章节 > 付费课程。讲义和带代码的材料优先于纯理论综述。

---

## 2. 项目现状

### 2.1 代码库

- 语言:Python 3.10+(NumPy 1.26、SciPy 1.12)和 C++17(CMake + Ninja Multi-Config,Windows/MSVC)。
- 当前只实现**欧式看涨期权**(call),连续股息率 `q`。
- 已有的函数(C++ 和 Python 各一套,命名对应):
  - 解析:`Phi_real`(用 `erfc` 实现标准正态 CDF)、`delta_call`、`Gamma`、`bs_price_call`。
  - 有限差分(FD):`delta_forward` = [C(S+h) − C(S)]/h;`gamma_forward` = [C(S+2h) − 2C(S+h) + C(S)]/h²。
  - Complex-step(CS):`delta_cs`(Im[C(S+ih)]/h)、`gamma_cs_real`(−2/h²·(Re C(S+ih) − C(S)))、`gamma_45_deg`(沿 (1+i)/√2 方向的对称位移,Im[C(S+ωh)+C(S−ωh)]/h²)。
  - 步长路由器 `route_hrel`:按 τ=σ√T、moneyness、ATM+小σ 把相对步长 `h_rel` 分段选在 [1e−12, 2e−7]。
  - 并行:C++ 的 `*_par` 版本用 `std::execution::par_unseq`。
- 数值稳定性措施:`log(F/K)` 在 F≈K 时用 `log1p`;φ(d1) 在对数空间计算;σ√T=0 时价格退化为折现内在价值、d1 取 ±∞。
- 测试:Python 9 个测试通过;C++ 有 `Phi_real_test`、`bs_greeks_test`(含 40,000 点压力测试)、新增的 `gamma_edge_test`。
- 一份已有的实验结论(S=K=100, r=q=0):
  - S1(σ=20%, T=1):Delta 误差中位数——FD 5.8e−6,CS 2.2e−16。Gamma 误差中位数——FD 0.199,CS-real 0.0198,CS-45° 9.3e−9。
  - S2(σ=1%, T=1/365,σ√T≈5.2e−4):Γ≈7.6,数值方法更脆弱。前向 FD 的 Γ 最大误差达 1.4e+14(灾难性抵消)。
  - 结论:Delta 用 CS,Gamma 用 45° CS,避免在生产中用 FD 的 Γ。

### 2.2 已修的 bug(背景,不需要你再处理)

C++ 标量 `Gamma` 的守卫写成 `if (T < 0 || S < 0 || sigma < 0) return 0;`,漏掉了等于 0 的情况。输入 `σ=0`、`T=0` 或 `S=0` 时,`sT = σ√T = 0`,于是 `d1 = (ln + 0.5σ²T)/sT = 0/0 = NaN`,标量和向量版返回 NaN;而并行版 `Gamma_par` 的守卫是 `<= 0`,返回 0。同一个库里三个入口两种答案。已改为 `<= 0.0` 并加回归测试。

教训:**退化输入(σ√T→0)的语义必须在一个地方定义,而不是每个入口各写各的。** 这正是下面重构的动机。

### 2.3 已识别的代码问题

- `delta_forward`、`gamma_forward`、`bs_price_call` 各自把 `DF, F, σ√T, ln(F/K), d1, d2` 的计算复制了 3–4 遍。
- Complex-step 的价格函数只存在于测试文件里,没进库。
- Python `delta_call` 在 F=K 且 σ√T=0 时 `np.sign(0)*inf = NaN`,C++ 同样输入返回 `0.5·e^{−qT}`,不一致。
- `log_F_over_K` 的 `log1p` 阈值是 `|x| ≤ 1e-12`,实际几乎不触发,这个设计的收益需要重新评估(见 6.2)。
- 有一个拼写错误的兼容别名 `detla_forward`。

---

## 3. 重构与扩展计划(设计已定,请审查)

**范围**:C++ 为主战场,Python 做对照/参考实现。不做:二叉树、美式期权、Monte Carlo(以后再说)。

### 步骤 1:单一内核(行为保持不变的重构)

- 新增内部函数 `bs_terms(S,K,r,q,σ,T)`,返回 `{DF, F, sT, d1, d2, degenerate}`。
- 所有路径(标量、向量、并行、FD)都从它出发,退化输入的处理收敛到一处。
- `_par` 里重复的"`par_unseq` + 指针算下标"样板抽成一个模板 helper。
- 现有公开函数保留为薄封装;重构前后 `bs_greeks_test` 的输出必须一致。

### 步骤 2:Put 与 put-call parity

- `enum class OptionType { Call, Put }`,统一的 `price(type, …)`。
- 测试:put-call parity;put 的各 Greek 对照 CS。

### 步骤 3:完整 Greeks(Delta、Gamma、Vega、Theta、Rho)

- 一次计算共享 d1/d2,返回 `struct Greeks`。
- 约定(默认):Vega 按每 1.00 波动率;Theta 按每年且为对**日历时间 t** 的导数(即 −∂V/∂T);Rho 按每 1.00 利率。
- 把 complex-step 价格函数(`std::complex<double>`)从测试搬进库,用它校验所有解析 Greeks。
- 退化输入约定:Gamma 与 Vega 在 S/σ/T ≤ 0 时为 0;价格为折现内在价值;C++/Python 统一。

### 步骤 4:隐含波动率

- `implied_vol(type, price, S,K,r,q,T)` → `{vol, status}`。
- 先做无套利边界检查,越界返回 `PriceOutOfBounds` 而不是 NaN。
- 算法:带 bracket 的安全牛顿法,牛顿步出界或 Vega 过小时退回二分,容差 1e−12。
- 测试:在 (K/S, σ, T) 网格上做 price→IV→price 往返,覆盖深度 ITM/OTM 与 T=1/365。

### 步骤 5:Python 镜像 + golden 文件

- Python 补齐 put、Greeks、`implied_vol` 作为参考实现。
- Python 生成一份 golden CSV(输入与期望值),提交进仓库,C++ 测试读它做对照,防止两种语言悄悄漂移。

每一步 TDD,每步结束时所有既有测试仍须通过。

---

## 4. 需要我理解的数学(请据此找材料)

约定:`F = S·e^{(r−q)T}`,`DF = e^{−rT}`,`φ`/`Φ` 为标准正态 PDF/CDF。

```
d1 = ( ln(F/K) + ½σ²T ) / (σ√T),   d2 = d1 − σ√T

Call = DF · ( F·Φ(d1) − K·Φ(d2) )
Put  = DF · ( K·Φ(−d2) − F·Φ(−d1) )
Parity:  C − P = S·e^{−qT} − K·e^{−rT}

Delta:  call  e^{−qT}Φ(d1)          put  −e^{−qT}Φ(−d1)
Gamma:  e^{−qT} φ(d1) / (S σ √T)                (call = put)
Vega:   S e^{−qT} φ(d1) √T                      (call = put)
Theta (−∂V/∂T):
        call  −S e^{−qT} φ(d1) σ /(2√T) − r K e^{−rT} Φ(d2) + q S e^{−qT} Φ(d1)
        put   −S e^{−qT} φ(d1) σ /(2√T) + r K e^{−rT} Φ(−d2) − q S e^{−qT} Φ(−d1)
Rho:    call  K T e^{−rT} Φ(d2)     put  −K T e^{−rT} Φ(−d2)
```

请确认以上公式无误(尤其 Theta 的符号与 q 项);有错请指出。

需要覆盖的数学主题:

1. **BS 公式的两条推导路线**:PDE(Black–Scholes 方程 + 边界条件)与风险中性定价(Girsanov、鞅测度)。我需要两条都能讲清,并理解 `F`/远期形式为什么让公式更简洁。
2. **Greeks 的含义与关系**:Delta/Gamma/Theta 之间的 BS PDE 关系(Θ + ½σ²S²Γ + (r−q)SΔ − rV = 0);Vega 与 Gamma 的关系(Vanna–Volga 的前置知识);各 Greek 随 moneyness、T 的形状;为什么 ATM 近到期时 Γ 会爆炸。
3. **Put-call parity** 的无套利推导,及它在测试与 IV 边界里的用法。
4. **隐含波动率的数学**:
   - 价格对 σ 的单调性(Vega>0),所以 IV 唯一;
   - 价格关于 σ 的凸/凹结构(存在拐点),这是牛顿法从糟糕初值出发会发散的原因;
   - 无套利价格边界;
   - 深度 OTM 或短期限时 Vega→0 的病态性(条件数)。
5. **复变量求导(complex-step)的理论**:为什么 `f'(x) ≈ Im f(x+ih)/h` 没有减法抵消;误差为 O(h²);二阶导为什么不能直接用单次 CS,而 45° 对称位移如何把误差降到 O(h⁴);它对"解析函数"和"函数内部必须逐运算符支持复数"的要求(例如 `abs`、比较运算会破坏它)。
6. **有限差分的误差理论**:截断误差 vs 舍入误差,一阶导的最优步长 h* ~ ε^{1/2}(前向差分)、ε^{1/3}(中心差分);二阶导的前向三点公式截断误差 O(h)、舍入误差 ~ε/h²,h* ~ ε^{1/3};中心二阶差分 h* ~ ε^{1/4}。仓库里的 `gamma_forward` 是前向三点公式。

---

## 5. 需要我理解的数值工程

1. **浮点算术基础**:IEEE 754 双精度、机器精度 ε≈2.2e−16、舍入误差模型、灾难性抵消、`NaN`/`±inf` 的传播规则、`0/0` 与 `x/0`。
2. **条件数与稳定性**:问题的条件数 vs 算法的向后稳定性;为什么 FD 的 Γ 在小 h 下炸掉,而 CS 的 Γ 不会。
3. **标准正态 CDF 的实现**:为什么用 `erfc` 而不是 `1 − erf`(尾部的相对精度);Cody 算法;`Φ(−z)` 与 `1−Φ(z)` 的精度差异。
4. **`log1p` / `expm1` 一类的"稳定原语"**:什么时候它们真的有用(需要给出量化标准,而不是凭感觉)。
5. **根查找**:Newton、二分、割线、Brent 方法;safeguarded Newton 的写法;收敛判据(绝对 vs 相对容差、价格空间 vs 波动率空间)。
6. **隐含波动率的工业级做法**:Jäckel 的有理函数近似("Let's be rational")为何比牛顿法稳;初值选择(Brenner–Subrahmanyam、Corrado–Miller)。
7. **退化输入的设计**:σ√T→0 的极限行为;"同一个语义只在一处定义"的工程原则;用 `status` 枚举代替 NaN 的错误处理设计。
8. **C++ 性能与并行**:`std::execution::par_unseq` 的语义与限制(无锁、无异常、迭代间独立);用 `&slot - base` 算下标这种写法的风险;SIMD 自动向量化的前提;函数模板作为"零开销抽象"来去重。
9. **数值测试方法学**:容差怎么选(相对 vs 绝对,按条件数缩放);golden file 测试;property-based 测试(parity、单调性、对称性);对照第二实现(Python vs C++)。

---

## 6. 几个我自己没想清楚的问题(请在材料里顺带回答)

1. 我的 `log1p` 阈值取 `|x| ≤ 1e−12`:在这个阈值下,`log(F/K)` 的误差到底有多大?`σ√T≈5e−4` 时这个误差经 `/σ√T` 放大后会不会成问题?阈值应该设在多少才合理?
2. `gamma_45_deg` 里我用了一个二阶泰勒的复数 `Phi_t`(只展开到 `b²`,虚部只到 `b¹`)。这个近似对 45° 公式的 O(h⁴) 声明是否仍然成立?更严谨的做法是什么(例如对复数 `erfc` 用 Faddeeva 函数 `w(z)`)?
3. 步长路由器 `route_hrel` 的分段阈值是经验值。有没有理论依据可以推出 `h_rel` 与 σ√T 的关系?
4. 对 IV:当 Vega 极小(深度 OTM、短期限)时,应该在价格空间还是波动率空间判收敛?
5. `par_unseq` 里用 `std::exp`、`std::erfc` 这类可能抛出或设置 errno 的函数,在标准里是否合法?

---

## 7. 材料检索要求

### 7.1 总体原则

- 目标是**让我真正理解并能在面试中讲清楚**,不是堆书单。宁可少而准。
- 每个主题给 **1 个首选 + 最多 2 个备选**。
- 免费/开放获取优先。注明是否需要付费或机构登录。

### 7.2 每条材料必须包含

| 字段 | 说明 |
|---|---|
| 主题 | 对应第 4/5 节的哪一条 |
| 对应计划步骤 | 步骤 1–5 中的哪一步最需要它 |
| 类型 | 教材章节 / 讲义 / 论文 / 博客 / 带代码仓库 / 视频课 |
| 完整出处 | 作者、年份、标题、出版方或期刊(卷期)、URL |
| 具体读哪里 | 章节或页范围;不确定就写"需核对" |
| 先修 | 读之前需要会什么 |
| 难度与耗时 | 初级/中级/高级;预计小时数 |
| 读完能做到 | 一两句话,可检验 |
| 验证状态 | 已验证链接 / 未验证;置信度 高/中/低 |

### 7.3 每个主题配 2–3 道练习

练习必须**和这个仓库绑定**,例如:

- "用 Python 画出 σ=1%、T=1/365 下 FD-Γ 误差随 h 的曲线,标出最优 h*,并和你推出的 ε^{1/3}(前向三点公式)比较;再换成中心二阶差分,对照 ε^{1/4}。"
- "证明 put 的 Delta = call 的 Delta − e^{−qT},并用 parity 数值验证。"
- "构造一个让牛顿法 IV 发散的 (K/S, σ, T, 初值),再说明 safeguarded 版本为什么能救回来。"

### 7.4 学习路径

给一份**按周**的路径(建议 4–5 周,每周约 8 小时),每周列出:读什么、做哪几道练习、和计划哪一步对齐。顺序要让我在写步骤 1 之前先具备浮点与稳定性基础,在步骤 4 之前先学完根查找与 IV 的数学。

### 7.5 种子清单(我凭记忆列的,未经验证,请逐条核实)

- Hull,*Options, Futures, and Other Derivatives*(BS 公式、Greeks、parity 的标准教材章节)。
- Shreve,*Stochastic Calculus for Finance II*(风险中性定价、Girsanov、BS 的鞅推导)。
- Goldberg,"What Every Computer Scientist Should Know About Floating-Point Arithmetic",ACM Computing Surveys,1991。
- Higham,*Accuracy and Stability of Numerical Algorithms*,SIAM,2002(舍入误差、条件数、稳定性)。
- Squire & Trapp,"Using Complex Variables to Estimate Derivatives of Real Functions",SIAM Review,1998。
- Martins, Sturdza, Alonso,"The Complex-Step Derivative Approximation",ACM Transactions on Mathematical Software,2003。
- Jäckel,"Let's Be Rational",Wilmott,2015(隐含波动率的近乎机器精度解法)。
- Brent,*Algorithms for Minimization without Derivatives*,1973(Brent 方法)。
- Cody,"Rational Chebyshev Approximations for the Error Function",Mathematics of Computation,1969。
- Wichura,"Algorithm AS 241: The Percentage Points of the Normal Distribution",1988(仓库里另有逆正态 CDF 实现,用到 Halley 精化)。
- Gatheral,*The Volatility Surface*(IV 的实务背景,后续扩展用)。
- Joshi,*C++ Design Patterns and Derivatives Pricing*(定价库的 C++ 设计)。
- cppreference 上关于 `std::execution` 执行策略的页面(`par_unseq` 对函数的限制)。
- Kahan 的数值分析讲义(浮点陷阱);Numerical Recipes 里根查找一章。

如果你有更好的、更新的、或开放获取的替代品,请直接换掉并说明理由。

---

## 8. 输出格式

按下面顺序回答:

1. **确认与校准**:对第 1 节我的水平假设有疑问就先提问;没有就写"按此校准"。
2. **计划审查**:逐条列出第 3–5 节里你认为有误或可疑的地方(公式符号、约定、算法、设计),分"必须改 / 建议改 / 小问题"。没有就明说没有,不要凑数。
3. **第 6 节问题的回答**:每题一段,给结论和依据。
4. **学习材料表**:按第 7.2 节字段,按主题分组。
5. **练习清单**:按第 7.3 节。
6. **按周学习路径**:按第 7.4 节。
7. **未验证项汇总**:列出所有你没能确认的链接或引用,方便我自己核对。
