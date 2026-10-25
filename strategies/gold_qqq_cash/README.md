# 子策略：现金 / 黄金 / QQQ

资产只有三类：**现金**（T-bill）、**黄金**（GLD）、**QQQ**（纳指 100）。资金来源是工资收入的一部分，按月定投。研究目标是找出新增资金在三者之间的分配规则（以及是否需要再平衡），让组合在 QQQ 大跌时有可靠的对冲。

## 目录

| 路径 | 内容 |
| --- | --- |
| `data_pipeline.py` | 简易数据管道：Yahoo/FRED 拉取 → 规范化 → 校验 → 缓存 → manifest |
| `data/manifest.json` | 每个序列的行数、日期范围、锚点值和校验问题（**入库**；用于发现供应商改写历史） |
| `data/raw/` | 原始 CSV 缓存（**不入库**，`.gitignore` 已忽略；运行管道即可重建） |
| `docs/data_pipeline_lessons.md` | **数据管道踩坑记录**：每个坑都附实测证据和处理方式 |
| `study_01_rolling_correlation.py` | 研究 01：黄金–QQQ 滚动相关性（20/60 日）与符号翻转 |
| `reports/01_rolling_correlation.md` | 研究 01 报告 |
| `artifacts/01_rolling_correlation/` | 研究 01 的图、CSV 和 `results.json` |
| `tests/` | 离线单元测试（不联网） |

## 运行

在仓库根目录执行：

```bash
python -m pip install -r strategies/gold_qqq_cash/requirements.txt
python strategies/gold_qqq_cash/data_pipeline.py                 # 拉取缺失或过期（>20h）的序列
python strategies/gold_qqq_cash/data_pipeline.py --refresh       # 强制全部重拉
python strategies/gold_qqq_cash/study_01_rolling_correlation.py  # 重新生成研究 01 的产物
python -m unittest discover -s strategies/gold_qqq_cash/tests
```

管道的数据范围见 `data_pipeline.UNIVERSE`：QQQ、GLD、IAU、GC=F（仅用于诊断）、^VIX、DX-Y.NYB、BIL，以及 FRED 的 DFII10（10 年期实际利率）和 DTB3（3 个月 T-bill）。

## 研究进度

| # | 题目 | 状态 | 一句话结论 |
| --- | --- | --- | --- |
| 01 | 黄金–QQQ 滚动相关性与符号翻转 | 完成 | 60 日相关性的状态是真实的（不是噪声），也能提前判断黄金在 QQQ 大跌日是否起对冲作用；20 日相关性的符号翻转大多是噪声 |
| 02 | 按相关性状态分配定投资金（只分配新增资金，不卖出） | 计划中 | — |
| 03 | 更平滑的相关性估计（EWMA / DCC），以及美元/实际利率因子模型 | 计划中 | — |
