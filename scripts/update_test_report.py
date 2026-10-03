"""Run current tests and regenerate an evidence-based Markdown report."""
from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import io
import json
import platform
import re
import subprocess
import sys
import traceback
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "test-report"
REPORT = ROOT / "docs" / "test_status.md"


def command(args):
    try:
        p = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=180)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def python_tests():
    records = []

    class Result(unittest.TextTestResult):
        def record(self, test, status, reason=""):
            records.append({"id": test.id(), "status": status, "reason": reason})

        def addSuccess(self, test):
            super().addSuccess(test)
            self.record(test, "PASS")

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.record(test, "FAIL", self._exc_info_to_string(err, test))

        def addError(self, test, err):
            super().addError(test, err)
            self.record(test, "ERROR", self._exc_info_to_string(err, test))

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.record(test, "SKIP", reason)

        def addExpectedFailure(self, test, err):
            super().addExpectedFailure(test, err)
            self.record(test, "XFAIL", self._exc_info_to_string(err, test))

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            self.record(test, "FAIL", "Unexpected success: remove or revise expectedFailure")

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err:
                self.record(test, "FAIL", self._exc_info_to_string(err, subtest))

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
    (OUT / "python.log").write_text(stream.getvalue(), encoding="utf-8")
    return records, result.wasSuccessful() and result.testsRun > 0


def cpp_tests(build_dir, config, build, generator):
    log = ""
    if build:
        for args in (["cmake", "-S", ROOT / "BS_cpp", "-B", build_dir, "-G", generator],
                     ["cmake", "--build", build_dir, "--config", config, "--parallel", "4"]):
            code, output = command(args)
            log += "$ " + " ".join(map(str, args)) + "\n" + output + "\n"
            if code:
                (OUT / "cpp.log").write_text(log, encoding="utf-8")
                return [{"id": "cpp:build", "status": "ERROR", "reason": output}], False
    _, discovery = command(["ctest", "--test-dir", build_dir, "-C", config, "--show-only=json-v1"])
    try:
        commands = {test["name"]: test.get("command", []) for test in json.loads(discovery)["tests"]}
    except (ValueError, KeyError):
        commands = {}
    xml = OUT / "ctest.xml"
    # Remove only the previous report file so it cannot be mistaken for this run.
    xml.unlink(missing_ok=True)
    code, output = command(["ctest", "--test-dir", build_dir, "-C", config,
                            "--output-on-failure", "--output-junit", xml])
    log += output
    records = []
    if xml.exists():
        for case in ET.parse(xml).getroot().iter("testcase"):
            failure = case.find("failure")
            skipped = case.find("skipped")
            body = case.findtext("system-out", "")
            status = "FAIL" if failure is not None else "SKIP" if skipped is not None else "PASS"
            reason = body if status != "PASS" else ""
            if failure is not None:
                reason = failure.get("message", "") + "\n" + (failure.text or "") + "\n" + body
            if "Process not started" in body or "BAD_COMMAND" in reason:
                status = "BLOCKED"
                # Get an OS-level explanation rather than guessing from CTest.
                argv = commands.get(case.get("name", ""), [])
                if argv:
                    direct_code, detail = command(argv)
                    if not detail:
                        detail = f"Direct launch exit code: {direct_code}"
                    reason += "\nDirect launch diagnostic: " + detail
                    log += "\n" + detail
            records.append({"id": "cpp:" + case.get("name", "unknown"), "status": status, "reason": reason})
    if not records:
        records = [{"id": "cpp:discovery", "status": "ERROR", "reason": output or "No CTest cases discovered"}]
    (OUT / "cpp.log").write_text(log, encoding="utf-8")
    return records, code == 0 and all(x["status"] == "PASS" for x in records)


def curves(order=3):
    for name in ("atm_errors.png", "near_expiry_errors.png", "atm_errors.csv", "near_expiry_errors.csv"):
        (OUT / name).unlink(missing_ok=True)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "Python_prototype"))
    import black_scholes as bs
    from bs_batch import evaluate_call

    summaries, images = [], []
    hrel = np.logspace(-16, -2, 100)
    for slug, sigma, expiry in [("atm", 0.2, 1.0), ("near_expiry", 0.01, 1 / 365)]:
        args = (100.0, 100.0, 0.0, 0.0, sigma, expiry)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rows = []
        for ax, greek, truth, methods in [
            (axes[0], "Delta", float(bs.delta_call(*args)), [bs.delta_forward, bs.delta_cs]),
            (axes[1], "Gamma", float(bs.Gamma(*args)), [bs.gamma_forward, bs.gamma_cs_real, bs.gamma_45_deg])]:
            for method in methods:
                with np.errstate(all="ignore"):
                    kwargs = {"order": order} if method in (bs.delta_cs, bs.gamma_cs_real, bs.gamma_45_deg) else {}
                    actual = np.array([float(method(*args, h=float(h * args[0]), **kwargs)) for h in hrel])
                errors = np.abs(actual - truth)
                finite = errors[np.isfinite(errors)]
                summaries.append({"scenario": slug, "method": method.__name__,
                                  "max_abs": float(np.max(finite)) if finite.size else None,
                                  "median_abs": float(np.median(finite)) if finite.size else None,
                                  "min_abs": float(np.min(finite)) if finite.size else None,
                                  "nonfinite": int(np.sum(~np.isfinite(errors)))})
                # Plot zeros at a display floor only; CSV and metrics keep actual zeros.
                ax.loglog(hrel, np.maximum(errors, 1e-18), label=method.__name__ + (f" P{order}" if kwargs else ""))
                rows.extend((greek, method.__name__, order if kwargs else "", h, args[0] * h, a, truth, e)
                            for h, a, e in zip(hrel, actual, errors))
            ax.set(title=f"{slug}: {greek}", xlabel="relative step h/S", ylabel="absolute error")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = OUT / f"{slug}_errors.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        images.append(path.name)
        with (OUT / f"{slug}_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["greek", "method", "cdf_order", "h_rel", "h", "actual", "reference", "abs_error"])
            writer.writerows(rows)

    with (ROOT / "testdata" / "bs_call_golden.csv").open(encoding="utf-8", newline="") as f:
        golden = list(csv.DictReader(f))
    result = evaluate_call(*(np.array([float(row[k]) for row in golden]) for k in ("S", "K", "r", "q", "sigma", "T")))
    gold = {k: float(np.max(np.abs(getattr(result, k) - np.array([float(row[k]) for row in golden]))))
            for k in ("price", "delta", "gamma")}
    return summaries, images, gold


def inventory(registry):
    for relative in registry["sources"]:
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        if relative.endswith(".py"):
            names = [n.name for n in ast.parse(source).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
        else:
            names = sorted(set(re.findall(r"^(?:double|void|CallResult|std::complex<double>)\s+(\w+)\s*\(", source, re.M)))
        for name in names:
            yield relative, name


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def greeks(name):
    if name.lower().startswith("delta"):
        return r"$\Delta$"
    if name.lower().startswith("gamma"):
        return r"$\Gamma$"
    if name.startswith("Phi"):
        return r"$\Phi$"
    if name == "phi":
        return r"$\phi$"
    if name.startswith("bs_price_call"):
        return "$C$"
    if name.startswith("evaluate_call"):
        return r"$C,\;\Delta,\;\Gamma$"
    return "—"


def function_table(registry, records):
    lines = ["| Greeks | 函数 | 相对路径 | 实现状态 | 本次关联测试 |",
             "| --- | --- | --- | --- | --- |"]
    for path, name in inventory(registry):
        patterns = registry["evidence"].get(f"{path}:{name}", [])
        matches = [r for r in records if any(fnmatch.fnmatchcase(r["id"], p) for p in patterns)]
        status = ", ".join(sorted({r["status"] for r in matches})) if matches else "未验证 / 待补关联测试"
        if patterns and not matches:
            status += "（登记测试未在本次发现）"
        lines.append(f"| {greeks(name)} | `{name}` | [{path}](../{path}) | 检测到定义 | {status} |")
    return lines


def planned_table(registry):
    return ["| 工作 | 从哪个相对路径继续 | 下一步 |", "| --- | --- | --- |"] + [
        f"| {cell(x['name'])} | [{x['path']}](../{x['path']}) | {cell(x['next'])} |"
        for x in registry["planned"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "out" / "test-report")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--order", type=int, choices=range(2, 7), default=3)
    parser.add_argument("--build", action="store_true", help="Configure and compile before CTest")
    parser.add_argument("--skip-cpp", action="store_true", help="Explicit Python-only report; C++ remains unverified")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / "scripts" / "test_report_registry.json").read_text(encoding="utf-8"))
    records, ok = [], True
    for name, action in [("python:runner", python_tests), ("cpp:runner", lambda: cpp_tests(args.build_dir.resolve(), args.config, args.build, args.generator))]:
        if name.startswith("cpp") and args.skip_cpp:
            records.append({"id": "cpp:runner", "status": "SKIP", "reason": "Explicit --skip-cpp"})
            (OUT / "cpp.log").write_text("Skipped in this run: --skip-cpp\n", encoding="utf-8")
            (OUT / "ctest.xml").unlink(missing_ok=True)
            continue
        try:
            found, passed = action()
            records.extend(found)
            ok = ok and passed
        except Exception:
            records.append({"id": name, "status": "ERROR", "reason": traceback.format_exc()})
            ok = False
    summaries, images, gold = [], [], {}
    try:
        summaries, images, gold = curves(order=args.order)
    except Exception:
        records.append({"id": "report:curves", "status": "ERROR", "reason": traceback.format_exc()})
        ok = False
    stamp = datetime.now(timezone.utc).isoformat()
    _, commit = command(["git", "rev-parse", "HEAD"])
    _, dirty = command(["git", "status", "--porcelain"])
    payload = {"generated_at": stamp, "commit": commit.strip(), "dirty": bool(dirty.strip()),
               "python": sys.version, "platform": platform.platform(), "cdf_order": args.order, "tests": records,
               "curve_metrics": summaries, "golden_max_abs": gold}
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 自动测试与开发进度报告", "", f"生成时间：{stamp}；commit：`{commit.strip()}`；工作区有改动：{bool(dirty.strip())}。",
             f"运行环境：`{sys.executable}`；Python `{platform.python_version()}`；`{platform.platform()}`。", "",
             f"本次误差扫描的 CDF 阶数：**P{args.order}**；默认 P3。测试套件会独立验证全部支持阶数，不随扫描参数改变。", "",
             "由 `scripts/update_test_report.py` 生成，请修改实现、测试或 `scripts/test_report_registry.json` 后重新运行，勿手改本文件。",
             "`tests/test_*.py` 与已注册的 CTest 测试会自动发现。函数扫描范围为 registry 的 sources 列表。函数与测试的关联由 registry 明确维护；检测到函数定义只表示已有实现入口，不证明实现完整。C++ 同名重载合并显示，关联测试不保证覆盖所有重载。", "",
             "## 函数实现与验证", "",
             r"符号：$\Delta$ 为 Delta，$\Gamma$ 为 Gamma；$C$ 为 Call 价格，$\Phi$ / $\phi$ 为标准正态分布函数 / 密度函数。辅助函数标为 —。", ""]
    lines += function_table(registry, records)
    lines += ["", "## 待完成事项（人工维护的范围）", ""]
    lines += planned_table(registry)
    lines += ["", "## 本次全部测试", "", "PASS=满足当前断言；FAIL=断言失败；ERROR=运行错误；BLOCKED=进程未启动；SKIP/XFAIL 均不算通过。", "",
              "| 测试 ID | 结果 |", "| --- | --- |"]
    lines += [f"| `{cell(r['id'])}` | {r['status']} |" for r in records]
    lines += ["", "原始记录：[JSON](../artifacts/test-report/results.json)、[Python 日志](../artifacts/test-report/python.log)、[C++ 日志](../artifacts/test-report/cpp.log)。", "", "## 失败与受阻原因", ""]
    problems = [r for r in records if r["status"] != "PASS"]
    if not problems:
        lines.append("本次已运行测试均通过。通过范围以本报告测试列表为限。")
    for r in problems:
        lines += [f"### {r['id']} — {r['status']}", "", "```text", r["reason"].replace("```", "~~~"), "```", ""]
        if "4551" in r["reason"] or "Application Control" in r["reason"] or "应用程序控制" in r["reason"]:
            lines.append("已确认原因：Windows Application Control 拒绝启动该可执行文件（系统错误 4551）。测试断言尚未执行，当前不能判断其数值结果；需要在允许执行该构建产物的环境中重新运行。")
        else:
            lines.append("原因边界：以上为测试/系统原始证据；未识别的根因需进一步排查，不能据此断言数值实现错误。")
    lines += ["", "## 误差曲线与原因", "",
              "以下曲线是本次重新计算的 Python 诊断，参考为解析 Delta/Gamma。它们与同库解析公式比较，不能替代独立高精度验证。横轴为 h/S；绝对误差为 |计算值−参考值|。绘图将零误差显示在 1e-18，CSV 保留原值。",
              "前向差分在大步长时受截断误差影响，极小步长时受浮点舍入和相消影响；当 S+h=S 时步长不可分辨。复数 CDF 为局部截断展开：order=2 的 45° Gamma 一般为二阶，order=3..6 在适用的小步长区间一般为四阶；仍有浮点舍入限制。对应测试分别检查了阶数与参数传递。", ""]
    lines += [f"![{name}](../artifacts/test-report/{name})" for name in images]
    lines += ["", "## 误差汇总", "", "Golden 独立参考样例的 Python 最大绝对误差：", "", "| 输出 | 最大绝对误差 |", "| --- | --- |"]
    lines += [f"| {k} | {v:.6e} |" for k, v in gold.items()]
    lines += ["", "步长扫描的误差范围如下；最大值包括故意选择的不可用步长，不是验收阈值。当前扫描 2 个 ATM 场景、每场景 100 个步长，未代表整个输入域或 C++ 精度。", "",
              "| 场景 | 方法 | 最小绝对误差 | 中位绝对误差 | 最大绝对误差 | 非有限结果数 |", "| --- | --- | --- | --- | --- | --- |"]
    for x in summaries:
        nums = [f"{x[k]:.6e}" if x[k] is not None else "无有限值" for k in ("min_abs", "median_abs", "max_abs")]
        lines.append(f"| {x['scenario']} | {x['method']} | {' | '.join(nums)} | {x['nonfinite']} |")
    lines += ["", "原始曲线数据：[ATM CSV](../artifacts/test-report/atm_errors.csv)、[近到期 CSV](../artifacts/test-report/near_expiry_errors.csv)。", "",
              "通过测试表示误差满足测试里指定的容差，不表示数学上零误差。当前 golden 验收采用按参考量级切换的绝对+相对容差，详情见 `tests/test_bs_batch.py` 和 `BS_cpp/tests/bs_batch_contract_test.cpp`。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {REPORT}")
    for r in records:
        print(r["status"], r["id"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
