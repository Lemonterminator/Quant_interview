# 子策略：现金 / 黄金 / QQQ

资产只有三类：**现金**（T-bill）、**黄金**（GLD）、**QQQ**（纳指 100）。资金来源是工资收入的一部分，按月定投。研究目标是找出新增资金在三者之间的分配规则（以及是否需要再平衡），让组合在 QQQ 大跌时有可靠的对冲。

## 目录

| 路径 | 内容 |
| --- | --- |
| `data_pipeline.py` | 简易数据管道：Yahoo/FRED 拉取 → 规范化 → 校验 → 缓存 → manifest |
| `data/manifest.json` | 每个序列的行数、日期范围、锚点值和校验问题（**入库**；用于发现供应商改写历史） |
| `data/raw/` | 原始 CSV 缓存（**不入库**，`.gitignore` 已忽略；运行管道即可重建） |
| `docs/data_pipeline_lessons.md` | **数据管道踩坑记录**：每个坑都附实测证据和处理方式 |
| `stats_utils.py` | 共用统计工具（只依赖 numpy/pandas）：Newey-West OLS、Wald 检验、方差比、ADF、半衰期、循环平移检验 |
| `study_01_rolling_correlation.py` | 研究 01：黄金–QQQ 滚动相关性（20/60 日）与符号翻转 |
| `study_02_ratio_momentum_reversion.py` | 研究 02：金/QQQ 比值的动量与均值回复 |
| `study_03_vix_gold_lead_lag.py` | 研究 03：VIX 变化与黄金次日收益的领先-滞后关系 |
| `reports/0N_*.md` | 各研究的报告（中文） |
| `artifacts/0N_*/` | 各研究的图、CSV 和 `results.json` |
| `tests/` | 离线单元测试（不联网） |

## 运行

在仓库根目录执行：

```bash
python -m pip install -r strategies/gold_qqq_cash/requirements.txt
python strategies/gold_qqq_cash/data_pipeline.py                 # 拉取缺失或过期（>20h）的序列
python strategies/gold_qqq_cash/data_pipeline.py --refresh       # 强制全部重拉
python strategies/gold_qqq_cash/data_pipeline.py VIX_1600        # 只更新 16:00 ET 的小时级 VIX（研究 03 用；缓存逐次累积）
python strategies/gold_qqq_cash/study_01_rolling_correlation.py  # 重新生成研究 01 的产物（02、03 同理）
python -m unittest discover -s strategies/gold_qqq_cash/tests
```

管道的数据范围见 `data_pipeline.UNIVERSE`：QQQ、GLD、IAU、GC=F（仅用于诊断）、^VIX、DX-Y.NYB、BIL，FRED 的 DFII10（10 年期实际利率）和 DTB3（3 个月 T-bill），以及 `VIX_1600`：用 ^VIX 小时 K 线取的 16:00 ET 读数（日线 ^VIX 收盘是 16:15，比 GLD/QQQ 晚 15 分钟）。Yahoo 只提供约 730 天的小时数据，所以这个缓存每次合并而不是替换，从 2024-01 开始积累。

## 研究进度

| # | 题目 | 状态 | 一句话结论 |
| --- | --- | --- | --- |
| 01 | 黄金–QQQ 滚动相关性与符号翻转 | 完成 | 60 日相关性的状态是真实的（不是噪声），也能提前判断黄金在 QQQ 大跌日是否起对冲作用；20 日相关性的符号翻转大多是噪声 |
| 02 | 金/QQQ 比值的动量与均值回复 | 完成 | 比值是单位根（半衰期约 12 年），1–12 个月的动量和反转都不显著，且前后两段方向相反；唯一显著的日频反转来自 QQQ 自身。黄金/QQQ 之间按固定比例分配，不做比值择时 |
| 03 | 谁领先谁：VIX 变化与黄金次日收益 | 完成 | 两个方向都没有可靠的领先关系，加入当天行情后 VIX 无增量信息；VIX 大跌后黄金次日上涨（+13bp）主要发生在隔夜且集中在 2004–2012 年，不可交易。VIX 不用于黄金择时 |
| 04 | 按相关性状态分配定投资金（只分配新增资金，不卖出），含阈值再平衡 | 计划中 | — |
| 05 | 更平滑的相关性估计（EWMA / DCC），以及美元/实际利率因子模型 | 计划中 | — |
