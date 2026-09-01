from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.optimize import brentq, minimize_scalar

QDIR = Path(__file__).resolve().parents[1]
RESULT_ROOT = QDIR.parent
ROOT = RESULT_ROOT.parent
sys.path.insert(0, str(ROOT))

from common.five_layer_model import (  # noqa: E402
    discharging_state_space, exact_linear_solution, layer_mass, rk45_linear_solution,
)

for folder in [QDIR / "results", QDIR / "data", QDIR / "figures", QDIR / "logs"]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(QDIR / "logs" / "run.log", mode="w", encoding="utf-8"), logging.StreamHandler()],
)
LOG = logging.getLogger("q2")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    cfg = json.loads((QDIR / "config" / "q2_config.json").read_text(encoding="utf-8"))
    q1 = json.loads((RESULT_ROOT / "问题1_参数辨识与预测" / "results" / "identified_parameters.json").read_text(encoding="utf-8"))
    K, UA = q1["K_W_per_K"], q1["UA_W_per_K"]
    rho, cp, R, H = 1800.0, 1500.0, 2.5, 8.0
    mass = layer_mass(R, H, rho)
    initial = np.array(cfg["initial_c"], dtype=float)
    dense_times = np.arange(0.0, cfg["duration_s"] + cfg["dense_step_s"], cfg["dense_step_s"])

    def simulate(q, times=dense_times, k=K, ua=UA):
        A, b = discharging_state_space(q, k, ua, mass, cp, cfg["inlet_c"], cfg["ambient_c"])
        return exact_linear_solution(A, b, initial, times)

    def dense_margins(q, k=K, ua=UA, times=dense_times):
        T = simulate(q, times, k, ua)
        top_margin_series = T[:, 0] - cfg["top_temperature_min_c"]
        differences = np.abs(np.diff(T, axis=1))
        diff_margin_series = cfg["adjacent_difference_max_c"] - np.max(differences, axis=1)
        return float(np.min(top_margin_series)), float(np.min(diff_margin_series)), T

    def refined_margins(q, k=K, ua=UA):
        base_temp, base_diff, T = dense_margins(q, k, ua)
        # 在每个2s网格局部极值附近用标量有界搜索加密。
        top_idx = int(np.argmin(T[:, 0] - cfg["top_temperature_min_c"]))
        diffs = np.abs(np.diff(T, axis=1))
        flat_idx = int(np.argmax(diffs))
        time_idx, pair_idx = np.unravel_index(flat_idx, diffs.shape)

        def state_at(t):
            return simulate(q, np.array([float(t)]), k, ua)[0]

        lo = dense_times[max(0, top_idx - 1)]; hi = dense_times[min(len(dense_times) - 1, top_idx + 1)]
        top_res = minimize_scalar(lambda t: state_at(t)[0] - cfg["top_temperature_min_c"], bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        dlo = dense_times[max(0, time_idx - 1)]; dhi = dense_times[min(len(dense_times) - 1, time_idx + 1)]
        diff_res = minimize_scalar(lambda t: cfg["adjacent_difference_max_c"] - abs(state_at(t)[pair_idx] - state_at(t)[pair_idx + 1]), bounds=(dlo, dhi), method="bounded", options={"xatol": 1e-8})
        candidates_top = [(base_temp, dense_times[top_idx]), (float(top_res.fun), float(top_res.x))]
        candidates_diff = [(base_diff, dense_times[time_idx], pair_idx + 1), (float(diff_res.fun), float(diff_res.x), pair_idx + 1)]
        top_best = min(candidates_top, key=lambda x: x[0])
        diff_best = min(candidates_diff, key=lambda x: x[0])
        return {"top_margin_c": top_best[0], "top_time_s": top_best[1], "diff_margin_c": diff_best[0], "diff_time_s": diff_best[1], "diff_pair": int(diff_best[2])}

    def feasible(q):
        m = refined_margins(q)
        return min(m["top_margin_c"], m["diff_margin_c"]) >= 0.0

    def feasible_dense(q):
        a, b, _ = dense_margins(q)
        return min(a, b) >= 0.0

    # 必须先检查二分搜索的可行左端点。若q=0都不可行，则原约束可行集为空。
    zero_margins = refined_margins(0.0)
    if min(zero_margins["top_margin_c"], zero_margins["diff_margin_c"]) < 0:
        LOG.error("INFEASIBLE PRECHECK: q=0 margins=%s; bisection has no feasible lower bracket", zero_margins)
        records = []
        for run in range(3):
            start = time.perf_counter(); _ = refined_margins(0.0); elapsed = time.perf_counter() - start
            records.append({"算法": "可行性判断+二分搜索", "运行序号": run + 1, "状态": "失败：q=0即不可行，无法建立可行括区间", "最佳诊断流量_kg_s": 0.0, "最大最小裕量_C": min(zero_margins["top_margin_c"], zero_margins["diff_margin_c"]), "约束违反量_C": -min(zero_margins["top_margin_c"], zero_margins["diff_margin_c"]), "运行时间_s": elapsed, "设置": f"边界预检+{cfg['dense_step_s']}s网格+极值加密"})

        grid_best = None
        for run in range(3):
            start = time.perf_counter()
            coarse = np.arange(0.0, 100.0 + 0.05, 0.1)
            coarse_values = []
            for q in coarse:
                mt, md, _ = dense_margins(float(q)); coarse_values.append(min(mt, md))
            coarse_q = float(coarse[int(np.argmax(coarse_values))])
            fine = np.arange(max(0.0, coarse_q - 0.1), min(100.0, coarse_q + 0.1) + 0.00025, 0.0005)
            fine_values = []
            for q in fine:
                mt, md, _ = dense_margins(float(q)); fine_values.append(min(mt, md))
            q_best = float(fine[int(np.argmax(fine_values))])
            m_best = refined_margins(q_best); margin_best = min(m_best["top_margin_c"], m_best["diff_margin_c"])
            elapsed = time.perf_counter() - start
            grid_best = (q_best, m_best)
            records.append({"算法": "两级高精度网格搜索", "运行序号": run + 1, "状态": "确认0—100 kg/s内无可行流量", "最佳诊断流量_kg_s": q_best, "最大最小裕量_C": margin_best, "约束违反量_C": max(0.0, -margin_best), "运行时间_s": elapsed, "设置": "粗网格0.1 kg/s；最佳点附近细网格0.0005 kg/s"})
            LOG.info("infeasible grid run=%d best q=%.8f margins=%s", run + 1, q_best, m_best)

        scalar_best = None
        for run in range(3):
            start = time.perf_counter()
            def neg_margin(q):
                mt, md, _ = dense_margins(float(q)); return -min(mt, md)
            opt = minimize_scalar(neg_margin, bounds=(0.0, 100.0), method="bounded", options={"xatol": 1e-7, "maxiter": 300})
            candidates = [(0.0, neg_margin(0.0)), (100.0, neg_margin(100.0)), (float(opt.x), float(opt.fun))]
            q_best, _ = min(candidates, key=lambda x: x[1])
            m_best = refined_margins(q_best); margin_best = min(m_best["top_margin_c"], m_best["diff_margin_c"])
            elapsed = time.perf_counter() - start
            scalar_best = (q_best, m_best)
            records.append({"算法": "标量最大裕量优化", "运行序号": run + 1, "状态": "最优裕量仍为负，确认不可行", "最佳诊断流量_kg_s": q_best, "最大最小裕量_C": margin_best, "约束违反量_C": max(0.0, -margin_best), "运行时间_s": elapsed, "设置": "bounded标量优化+端点比较；xatol=1e-7"})
            LOG.info("infeasible scalar run=%d best q=%.8f margins=%s", run + 1, q_best, m_best)

        comp = pd.DataFrame(records)
        comp.to_csv(QDIR / "results" / "algorithm_comparison_runs.csv", index=False, encoding="utf-8-sig")
        comp.groupby("算法", sort=False).agg(最佳诊断流量均值_kg_s=("最佳诊断流量_kg_s", "mean"), 最大最小裕量均值_C=("最大最小裕量_C", "mean"), 违反量均值_C=("约束违反量_C", "mean"), 运行时间均值_s=("运行时间_s", "mean")).reset_index().to_csv(QDIR / "results" / "algorithm_comparison_summary.csv", index=False, encoding="utf-8-sig")

        # q=0是最小冷盐扰动的诊断轨迹；它仍违反顶部温度约束，不能称为可行放热方案。
        verify_times = np.arange(0.0, cfg["duration_s"] + cfg["verification_step_s"], cfg["verification_step_s"])
        T = simulate(0.0, verify_times); diffs = np.abs(np.diff(T, axis=1))
        power_w = np.zeros_like(verify_times); cumulative_j = np.zeros_like(verify_times)
        trajectory = pd.DataFrame(T, columns=[f"T{i}_C" for i in range(1, 6)]); trajectory.insert(0, "time_s", verify_times)
        for i in range(4): trajectory[f"delta_T{i+1}{i+2}_C"] = diffs[:, i]
        trajectory["power_MW"] = 0.0; trajectory["cumulative_heat_GJ"] = 0.0
        trajectory.to_csv(QDIR / "results" / "diagnostic_q0_trajectory_0p5s.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

        # 独立RK45复算q=0。
        A0, b0 = discharging_state_space(0.0, K, UA, mass, cp, cfg["inlet_c"], cfg["ambient_c"])
        check_times = np.arange(0, cfg["duration_s"] + 10, 10.0)
        exact_check = exact_linear_solution(A0, b0, initial, check_times)
        start = time.perf_counter(); rk45 = rk45_linear_solution(A0, b0, initial, check_times, 1e-11, 1e-12); rk_time = time.perf_counter() - start
        solver_check = {"max_abs_difference_C": float(np.max(np.abs(rk45 - exact_check))), "runtime_s": rk_time, "rtol": 1e-11, "atol": 1e-12}

        # 可行性恢复诊断：分别令K=0、UA=0或二者同比缩放，求q=0时T1(3h)=515的临界比例。
        def end_top(k, ua):
            return float(simulate(0.0, np.array([cfg["duration_s"]]), k, ua)[0, 0])
        thresholds = {}
        thresholds["UA_scale_if_K_zero"] = brentq(lambda s: end_top(0.0, s * UA) - 515.0, 0.0, 1.0)
        thresholds["K_scale_if_UA_zero"] = brentq(lambda s: end_top(s * K, 0.0) - 515.0, 0.0, 1.0)
        thresholds["common_scale_K_and_UA"] = brentq(lambda s: end_top(s * K, s * UA) - 515.0, 0.0, 1.0)

        sensitivity = []
        for name, kval, uaval in [("K -5%", 0.95*K, UA), ("K +5%", 1.05*K, UA), ("UA -5%", K, 0.95*UA), ("UA +5%", K, 1.05*UA)]:
            mm = refined_margins(0.0, kval, uaval)
            sensitivity.append({"情景": name, **mm, "可行": min(mm["top_margin_c"], mm["diff_margin_c"]) >= 0})
        pd.DataFrame(sensitivity).to_csv(QDIR / "results" / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")

        result = {
            "status": "infeasible",
            "conclusion": "在问题1辨识参数与题面初始条件下，q=0也无法维持T1>=515℃至3h，因此不存在满足全部连续约束的恒定放热流量。",
            "q_max_kg_s": None,
            "diagnostic_best_q_kg_s": 0.0,
            "q0_margins": zero_margins,
            "q0_top_temperature_at_3h_c": float(T[-1, 0]),
            "q0_max_adjacent_difference_c": float(np.max(diffs)),
            "feasible_effective_heat": None,
            "diagnostic_q0_heat": {"J": 0.0, "GJ": 0.0, "MWh": 0.0, "note": "q=0仍不可行，不是题目要求的有效放热方案"},
            "parameter_recovery_thresholds": thresholds,
            "solver_cross_check": solver_check,
            "K_W_per_K": K, "UA_W_per_K": UA,
            "verification_step_s": cfg["verification_step_s"],
        }
        (QDIR / "results" / "q2_final_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        q_plot = np.linspace(0, 100, 201); margin_data = []
        for q in q_plot:
            mt, md, _ = dense_margins(float(q)); margin_data.append((mt, md))
        pd.DataFrame({"q_kg_s": q_plot, "top_margin_c": [x[0] for x in margin_data], "diff_margin_c": [x[1] for x in margin_data]}).to_csv(QDIR / "data" / "flow_margin_curve.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(7.8, 4.4)); ax.plot(q_plot, [x[0] for x in margin_data], label="顶部温度裕量"); ax.plot(q_plot, [x[1] for x in margin_data], label="相邻温差裕量"); ax.axhline(0, color="black", lw=.8); ax.set_xlabel("恒定放热流量 / (kg/s)"); ax.set_ylabel("最小约束裕量 / ℃"); ax.set_title("问题2可行性诊断：全区间无可行流量"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig1_flow_feasibility.png", dpi=220); plt.close(fig)
        time_h = verify_times / 3600
        fig, ax = plt.subplots(figsize=(8.2, 4.6)); [ax.plot(time_h, T[:, i], label=f"T{i+1}") for i in range(5)]; ax.axhline(515, color="#DC2626", ls="--", label="出口下限515℃"); ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / ℃"); ax.set_title("最有利诊断情形q=0的五层温度"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig2_temperatures.png", dpi=220); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.2, 4.4)); [ax.plot(time_h, diffs[:, i], label=f"|T{i+1}-T{i+2}|") for i in range(4)]; ax.axhline(150, color="#DC2626", ls="--", label="上限150℃"); ax.set_xlabel("时间 / h"); ax.set_ylabel("相邻层温差 / ℃"); ax.set_title("q=0时相邻层温差检查"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig3_adjacent_differences.png", dpi=220); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.3, 4.2)); labels = [x["情景"] for x in sensitivity]; vals = [x["top_margin_c"] for x in sensitivity]; ax.bar(labels, vals, color=["#60A5FA", "#2563A6", "#FBBF24", "#D97706"]); ax.axhline(0, color="black"); ax.set_ylabel("q=0时顶部最小裕量 / ℃"); ax.set_title("K与UA的±5%灵敏度：仍全部不可行"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig4_sensitivity.png", dpi=220); plt.close(fig)
        LOG.error("FINAL INFEASIBLE: q=0 top_margin=%.6fC top_end=%.6fC thresholds=%s", zero_margins["top_margin_c"], T[-1,0], thresholds)
        return

    records = []
    bisect_values = []
    for run in range(3):
        start = time.perf_counter(); lo, hi = cfg["flow_bounds_kg_s"]
        assert feasible(lo) and not feasible(hi)
        iterations = 0
        while hi - lo > cfg["bisection_tolerance_kg_s"]:
            mid = (lo + hi) / 2.0
            if feasible(mid): lo = mid
            else: hi = mid
            iterations += 1
        elapsed = time.perf_counter() - start
        margins = refined_margins(lo)
        bisect_values.append(lo)
        records.append({"算法": "可行性判断+二分搜索", "运行序号": run + 1, "q_max_kg_s": lo, **margins, "约束违反量": max(0.0, -min(margins["top_margin_c"], margins["diff_margin_c"])), "运行时间_s": elapsed, "迭代次数": iterations, "设置": f"q容差={cfg['bisection_tolerance_kg_s']} kg/s；时间网格={cfg['dense_step_s']} s+极值加密"})
        LOG.info("bisection run=%d q=%.10f margins=%s time=%.4f", run + 1, lo, margins, elapsed)

    grid_values = []
    for run in range(3):
        start = time.perf_counter()
        coarse = np.arange(cfg["flow_bounds_kg_s"][0], cfg["flow_bounds_kg_s"][1] + cfg["grid_coarse_step_kg_s"] / 2, cfg["grid_coarse_step_kg_s"])
        feasible_coarse = [q for q in coarse if feasible_dense(float(q))]
        base = max(feasible_coarse)
        fine = np.arange(base, min(cfg["flow_bounds_kg_s"][1], base + cfg["grid_coarse_step_kg_s"] + cfg["grid_fine_step_kg_s"] / 2), cfg["grid_fine_step_kg_s"])
        feasible_fine = [q for q in fine if feasible_dense(float(q))]
        q_grid = float(max(feasible_fine))
        elapsed = time.perf_counter() - start
        margins = refined_margins(q_grid)
        grid_values.append(q_grid)
        records.append({"算法": "两级高精度网格搜索", "运行序号": run + 1, "q_max_kg_s": q_grid, **margins, "约束违反量": max(0.0, -min(margins["top_margin_c"], margins["diff_margin_c"])), "运行时间_s": elapsed, "迭代次数": len(coarse) + len(fine), "设置": f"粗网格={cfg['grid_coarse_step_kg_s']} kg/s；细网格={cfg['grid_fine_step_kg_s']} kg/s；时间极值加密"})
        LOG.info("grid run=%d q=%.10f margins=%s time=%.4f", run + 1, q_grid, margins, elapsed)

    comp = pd.DataFrame(records)
    comp.to_csv(QDIR / "results" / "algorithm_comparison_runs.csv", index=False, encoding="utf-8-sig")
    summary = comp.groupby("算法", sort=False).agg(q均值_kg_s=("q_max_kg_s", "mean"), q标准差=("q_max_kg_s", "std"), 顶部最小裕量均值_C=("top_margin_c", "mean"), 温差最小裕量均值_C=("diff_margin_c", "mean"), 运行时间均值_s=("运行时间_s", "mean"), 违反量最大值=("约束违反量", "max")).reset_index()
    summary.to_csv(QDIR / "results" / "algorithm_comparison_summary.csv", index=False, encoding="utf-8-sig")

    q_final = min(bisect_values) - 1e-6  # 向可行侧保留微小数值余量。
    verify_times = np.arange(0.0, cfg["duration_s"] + cfg["verification_step_s"], cfg["verification_step_s"])
    T = simulate(q_final, verify_times)
    diffs = np.abs(np.diff(T, axis=1))
    top_margin_series = T[:, 0] - cfg["top_temperature_min_c"]
    diff_margin_matrix = cfg["adjacent_difference_max_c"] - diffs
    top_idx = int(np.argmin(top_margin_series)); diff_flat = int(np.argmin(diff_margin_matrix)); dti, dpi = np.unravel_index(diff_flat, diff_margin_matrix.shape)
    final_margins = refined_margins(q_final)

    # 有效热量：0.5s网格Simpson，另用梯形积分交叉验证。
    power_w = q_final * cp * (T[:, 0] - cfg["inlet_c"])
    heat_j = float(simpson(power_w, verify_times))
    heat_trap_j = float(np.trapz(power_w, verify_times))
    cumulative_j = np.r_[0.0, cumulative_trapezoid(power_w, verify_times)]
    trajectory = pd.DataFrame(T, columns=[f"T{i}_C" for i in range(1, 6)])
    trajectory.insert(0, "time_s", verify_times)
    for i in range(4): trajectory[f"delta_T{i+1}{i+2}_C"] = diffs[:, i]
    trajectory["power_MW"] = power_w / 1e6
    trajectory["cumulative_heat_GJ"] = cumulative_j / 1e9
    trajectory.to_csv(QDIR / "results" / "final_trajectory_0p5s.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

    # RK45独立复算。
    A, b = discharging_state_space(q_final, K, UA, mass, cp, cfg["inlet_c"], cfg["ambient_c"])
    check_times = np.arange(0, cfg["duration_s"] + 10, 10.0)
    exact_check = exact_linear_solution(A, b, initial, check_times)
    start = time.perf_counter(); rk45 = rk45_linear_solution(A, b, initial, check_times, 1e-11, 1e-12); rk_time = time.perf_counter() - start
    solver_check = {"RK45_rtol": 1e-11, "RK45_atol": 1e-12, "grid_s": 10.0, "max_abs_difference_C": float(np.max(np.abs(rk45 - exact_check))), "runtime_s": rk_time}

    # 参数灵敏度：K、UA分别±5%，重新求最大可行流量。
    sensitivity = []
    for name, base_val in [("K", K), ("UA", UA)]:
        for delta in [-0.05, 0.05]:
            k_val, ua_val = K, UA
            if name == "K": k_val *= 1 + delta
            else: ua_val *= 1 + delta
            def feas_local(q):
                mm = refined_margins(q, k_val, ua_val)
                return min(mm["top_margin_c"], mm["diff_margin_c"]) >= 0
            lo, hi = cfg["flow_bounds_kg_s"]
            for _ in range(35):
                mid = (lo + hi) / 2
                if feas_local(mid): lo = mid
                else: hi = mid
            sensitivity.append({"参数": name, "相对变化": delta, "q_max_kg_s": lo, "相对标称变化_pct": 100 * (lo / q_final - 1), **refined_margins(lo, k_val, ua_val)})
    sens_df = pd.DataFrame(sensitivity)
    sens_df.to_csv(QDIR / "results" / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")

    result = {
        "q_max_kg_s": q_final,
        "selected_method": "可行性判断+二分搜索（两级网格独立验证）",
        "K_W_per_K": K, "UA_W_per_K": UA,
        "constraints": {"top_temperature_min_c": cfg["top_temperature_min_c"], "adjacent_difference_max_c": cfg["adjacent_difference_max_c"]},
        "final_margins": final_margins,
        "active_constraint": "顶部出口温度下限" if final_margins["top_margin_c"] < final_margins["diff_margin_c"] else "相邻层温差上限",
        "active_time_s": final_margins["top_time_s"] if final_margins["top_margin_c"] < final_margins["diff_margin_c"] else final_margins["diff_time_s"],
        "heat": {"J": heat_j, "MJ": heat_j / 1e6, "GJ": heat_j / 1e9, "kWh": heat_j / 3.6e6, "MWh": heat_j / 3.6e9, "simpson_minus_trapezoid_J": heat_j - heat_trap_j},
        "critical_dense_grid": {"top_min_c": float(T[top_idx, 0]), "top_time_s": float(verify_times[top_idx]), "max_adjacent_difference_c": float(diffs[dti, dpi]), "difference_time_s": float(verify_times[dti]), "difference_pair": int(dpi + 1)},
        "solver_cross_check": solver_check,
        "verification_step_s": cfg["verification_step_s"],
    }
    (QDIR / "results" / "q2_final_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可行性边界图。
    q_plot = np.linspace(max(0, q_final - 8), min(100, q_final + 8), 121)
    margins_plot = [refined_margins(float(q)) for q in q_plot]
    pd.DataFrame({"q_kg_s": q_plot, "top_margin_c": [m["top_margin_c"] for m in margins_plot], "diff_margin_c": [m["diff_margin_c"] for m in margins_plot]}).to_csv(QDIR / "data" / "flow_margin_curve.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.8, 4.4)); ax.plot(q_plot, [m["top_margin_c"] for m in margins_plot], label="顶部温度裕量"); ax.plot(q_plot, [m["diff_margin_c"] for m in margins_plot], label="相邻温差裕量"); ax.axhline(0, color="black", lw=0.8); ax.axvline(q_final, color="#DC2626", ls="--", label=f"q*={q_final:.3f} kg/s"); ax.set_xlabel("恒定放热流量 / (kg/s)"); ax.set_ylabel("最小约束裕量 / ℃"); ax.set_title("流量可行性边界"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig1_flow_feasibility.png", dpi=220); plt.close(fig)

    time_h = verify_times / 3600
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); [ax.plot(time_h, T[:, i], label=f"T{i+1}") for i in range(5)]; ax.axhline(515, color="#DC2626", ls="--", label="出口下限515℃"); ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / ℃"); ax.set_title("最大恒定流量下五层温度"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig2_temperatures.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4)); [ax.plot(time_h, diffs[:, i], label=f"|T{i+1}-T{i+2}|") for i in range(4)]; ax.axhline(150, color="#DC2626", ls="--", label="上限150℃"); ax.set_xlabel("时间 / h"); ax.set_ylabel("相邻层温差 / ℃"); ax.set_title("相邻层温差连续检查"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig3_adjacent_differences.png", dpi=220); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.2, 4.5)); ax1.plot(time_h, power_w / 1e6, color="#2563A6", label="瞬时有效功率"); ax1.set_xlabel("时间 / h"); ax1.set_ylabel("功率 / MW", color="#2563A6"); ax2 = ax1.twinx(); ax2.plot(time_h, cumulative_j / 1e9, color="#D97706", label="累计有效热量"); ax2.set_ylabel("累计热量 / GJ", color="#D97706"); ax1.grid(alpha=.25); ax1.set_title("瞬时功率与累计有效热量"); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig4_power_heat.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2)); labels = [f"{r['参数']} {r['相对变化']:+.0%}" for r in sensitivity]; vals = [r["q_max_kg_s"] for r in sensitivity]; ax.bar(labels, vals, color=["#60A5FA", "#2563A6", "#FBBF24", "#D97706"]); ax.axhline(q_final, color="black", ls="--", label="标称q*"); ax.set_ylabel("最大可行流量 / (kg/s)"); ax.set_title("K与UA对最大流量的灵敏度"); ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(QDIR / "figures" / "fig5_sensitivity.png", dpi=220); plt.close(fig)

    LOG.info("FINAL q=%.10f kg/s active=%s time=%.6fs margins=%s", q_final, result["active_constraint"], result["active_time_s"], final_margins)
    LOG.info("HEAT %.6f GJ = %.6f MWh; solver max diff %.3e C", heat_j / 1e9, heat_j / 3.6e9, solver_check["max_abs_difference_C"])


if __name__ == "__main__":
    main()
