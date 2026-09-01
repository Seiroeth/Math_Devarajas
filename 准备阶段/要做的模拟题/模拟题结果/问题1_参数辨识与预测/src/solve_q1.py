from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize


QUESTION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUESTION_DIR.parent
sys.path.insert(0, str(ROOT))

from common.five_layer_model import (  # noqa: E402
    charging_state_space,
    energy_balance_residual,
    exact_linear_solution,
    layer_mass,
    rk4_linear_solution,
    rk45_linear_solution,
)


CONFIG_PATH = QUESTION_DIR / "config" / "q1_config.json"
RESULTS = QUESTION_DIR / "results"
DATA = QUESTION_DIR / "data"
FIGURES = QUESTION_DIR / "figures"
LOGS = QUESTION_DIR / "logs"
for folder in (RESULTS, DATA, FIGURES, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGS / "run.log", mode="w", encoding="utf-8"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("q1")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def metrics(residual: np.ndarray) -> dict[str, float]:
    flat = np.asarray(residual, dtype=float).ravel()
    return {
        "SSE": float(flat @ flat),
        "RMSE_C": float(np.sqrt(np.mean(flat**2))),
        "MAE_C": float(np.mean(np.abs(flat))),
        "MaxAbsError_C": float(np.max(np.abs(flat))),
    }


def main() -> None:
    cfg = load_config()
    geo, prop, ident, pred, num = (cfg[k] for k in ["geometry", "properties", "identification", "prediction", "numerics"])
    mass = layer_mass(geo["radius_m"], geo["height_m"], prop["density_kg_m3"], geo["layers"])
    heat_capacity = mass * prop["cp_j_kg_k"]
    times_s = np.asarray(ident["times_h"], dtype=float) * 3600.0
    observations = np.asarray(ident["observations_c"], dtype=float)
    observed_idx = np.asarray(ident["observed_layers"], dtype=int) - 1
    initial = np.asarray(ident["initial_c"], dtype=float)
    lower = np.array([ident["bounds"]["K_w_k"][0], ident["bounds"]["UA_w_k"][0]], dtype=float)
    upper = np.array([ident["bounds"]["K_w_k"][1], ident["bounds"]["UA_w_k"][1]], dtype=float)

    LOGGER.info("每层质量 %.10f kg；每层热容 %.10f J/K", mass, heat_capacity)
    LOGGER.info("统一参数边界 K=[%.1f, %.1f] W/K, UA=[%.1f, %.1f] W/K", lower[0], upper[0], lower[1], upper[1])

    def simulate(params: np.ndarray, eval_times: np.ndarray = times_s) -> np.ndarray:
        K, UA = np.asarray(params, dtype=float)
        A, b = charging_state_space(
            ident["flow_kg_s"], K, UA, mass, prop["cp_j_kg_k"], ident["inlet_c"], prop["ambient_c"]
        )
        return exact_linear_solution(A, b, initial, eval_times)

    def residual_vector(params: np.ndarray, target: np.ndarray = observations) -> np.ndarray:
        return (simulate(params)[:, observed_idx] - target).ravel()

    def objective(params: np.ndarray, target: np.ndarray = observations) -> float:
        r = residual_vector(params, target)
        return float(r @ r)

    records: list[dict] = []
    solutions: dict[str, list[np.ndarray]] = {"非线性最小二乘": [], "差分进化": [], "网格+局部优化": []}

    for run_idx, seed in enumerate(ident["seeds"], 1):
        rng = np.random.default_rng(seed)
        x0 = lower + rng.random(2) * (upper - lower)
        start = time.perf_counter()
        ls = least_squares(
            residual_vector,
            x0=x0,
            bounds=(lower, upper),
            xtol=num["least_squares_xtol"],
            ftol=num["least_squares_ftol"],
            gtol=num["least_squares_gtol"],
            max_nfev=3000,
        )
        elapsed = time.perf_counter() - start
        met = metrics(residual_vector(ls.x))
        records.append({"算法": "非线性最小二乘", "运行序号": run_idx, "随机种子": seed, "K_W_per_K": ls.x[0], "UA_W_per_K": ls.x[1], **met, "运行时间_s": elapsed, "成功": bool(ls.success), "迭代信息": ls.message})
        solutions["非线性最小二乘"].append(ls.x.copy())
        LOGGER.info("LS run=%d seed=%d K=%.8f UA=%.8f SSE=%.10f time=%.4fs", run_idx, seed, ls.x[0], ls.x[1], met["SSE"], elapsed)

        start = time.perf_counter()
        de = differential_evolution(
            objective,
            bounds=list(zip(lower, upper)),
            seed=seed,
            tol=num["differential_evolution_tol"],
            popsize=num["differential_evolution_popsize"],
            maxiter=num["differential_evolution_maxiter"],
            polish=True,
            updating="immediate",
            workers=1,
        )
        elapsed = time.perf_counter() - start
        met = metrics(residual_vector(de.x))
        records.append({"算法": "差分进化", "运行序号": run_idx, "随机种子": seed, "K_W_per_K": de.x[0], "UA_W_per_K": de.x[1], **met, "运行时间_s": elapsed, "成功": bool(de.success), "迭代信息": de.message})
        solutions["差分进化"].append(de.x.copy())
        LOGGER.info("DE run=%d seed=%d K=%.8f UA=%.8f SSE=%.10f time=%.4fs", run_idx, seed, de.x[0], de.x[1], met["SSE"], elapsed)

        start = time.perf_counter()
        K_grid = np.linspace(lower[0], upper[0], num["grid_K_points"])
        UA_grid = np.linspace(lower[1], upper[1], num["grid_UA_points"])
        best = (np.inf, None)
        for K in K_grid:
            for UA in UA_grid:
                value = objective(np.array([K, UA]))
                if value < best[0]:
                    best = (value, np.array([K, UA]))
        local = least_squares(
            residual_vector,
            x0=best[1],
            bounds=(lower, upper),
            xtol=num["least_squares_xtol"],
            ftol=num["least_squares_ftol"],
            gtol=num["least_squares_gtol"],
            max_nfev=3000,
        )
        elapsed = time.perf_counter() - start
        met = metrics(residual_vector(local.x))
        records.append({"算法": "网格+局部优化", "运行序号": run_idx, "随机种子": seed, "K_W_per_K": local.x[0], "UA_W_per_K": local.x[1], **met, "运行时间_s": elapsed, "成功": bool(local.success), "迭代信息": local.message})
        solutions["网格+局部优化"].append(local.x.copy())
        LOGGER.info("GRID+LOCAL run=%d K=%.8f UA=%.8f SSE=%.10f time=%.4fs", run_idx, local.x[0], local.x[1], met["SSE"], elapsed)

    comparison = pd.DataFrame(records)
    comparison.to_csv(RESULTS / "algorithm_comparison_runs.csv", index=False, encoding="utf-8-sig")
    summary = comparison.groupby("算法", sort=False).agg(
        K均值=("K_W_per_K", "mean"), K标准差=("K_W_per_K", "std"),
        UA均值=("UA_W_per_K", "mean"), UA标准差=("UA_W_per_K", "std"),
        SSE均值=("SSE", "mean"), RMSE均值_C=("RMSE_C", "mean"),
        最大绝对误差均值_C=("MaxAbsError_C", "mean"), 运行时间均值_s=("运行时间_s", "mean"),
        成功率=("成功", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS / "algorithm_comparison_summary.csv", index=False, encoding="utf-8-sig")

    # 三类算法已收敛至同一数值邻域；以耗时最短且便于协方差分析的非线性最小二乘作为最终方法。
    ls_rows = comparison[comparison["算法"] == "非线性最小二乘"]
    best_row = ls_rows.loc[ls_rows["SSE"].idxmin()]
    final_params = np.array([best_row["K_W_per_K"], best_row["UA_W_per_K"]], dtype=float)
    final_fit = simulate(final_params)
    fit_residuals = final_fit[:, observed_idx] - observations
    final_metrics = metrics(fit_residuals)

    # Jacobian协方差近似、相关性和条件数。
    final_ls = least_squares(residual_vector, x0=final_params, bounds=(lower, upper), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
    jac = final_ls.jac
    dof = observations.size - 2
    sigma2 = float(np.sum(final_ls.fun**2) / dof)
    covariance = sigma2 * np.linalg.inv(jac.T @ jac)
    std = np.sqrt(np.diag(covariance))
    normal_ci = np.column_stack([final_params - 1.96 * std, final_params + 1.96 * std])
    correlation = covariance[0, 1] / (std[0] * std[1])
    scaled_jac = jac * final_params[np.newaxis, :]
    condition_number = float(np.linalg.cond(scaled_jac))

    # 残差Bootstrap：以21个标量残差为经验分布，固定种子并重新拟合。
    rng = np.random.default_rng(20260825)
    fitted_obs = final_fit[:, observed_idx]
    centered = fit_residuals.ravel() - np.mean(fit_residuals)
    bootstrap = []
    for rep in range(num["bootstrap_replicates"]):
        boot_target = fitted_obs + rng.choice(centered, size=centered.size, replace=True).reshape(fitted_obs.shape)
        def boot_res(params: np.ndarray) -> np.ndarray:
            return (simulate(params)[:, observed_idx] - boot_target).ravel()
        result = least_squares(boot_res, x0=final_params, bounds=(lower, upper), xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=800)
        bootstrap.append([rep + 1, result.x[0], result.x[1], float(result.fun @ result.fun), bool(result.success)])
    boot_df = pd.DataFrame(bootstrap, columns=["replicate", "K_W_per_K", "UA_W_per_K", "SSE", "success"])
    boot_df.to_csv(DATA / "bootstrap_parameters.csv", index=False, encoding="utf-8-sig")
    boot_ci = np.percentile(boot_df[["K_W_per_K", "UA_W_per_K"]], [2.5, 97.5], axis=0).T

    # ODE求解器交叉验证。
    A_fit, b_fit = charging_state_space(ident["flow_kg_s"], final_params[0], final_params[1], mass, prop["cp_j_kg_k"], ident["inlet_c"], prop["ambient_c"])
    solver_records = []
    start = time.perf_counter(); exact_fit = exact_linear_solution(A_fit, b_fit, initial, times_s); exact_time = time.perf_counter() - start
    solver_records.append({"求解器": "矩阵指数", "设置": "常系数增广矩阵expm", "运行时间_s": exact_time, "相对矩阵指数最大误差_C": 0.0})
    start = time.perf_counter(); rk45_fit = rk45_linear_solution(A_fit, b_fit, initial, times_s, num["rk45_rtol"], num["rk45_atol"]); rk45_time = time.perf_counter() - start
    solver_records.append({"求解器": "RK45", "设置": f"rtol={num['rk45_rtol']}, atol={num['rk45_atol']}", "运行时间_s": rk45_time, "相对矩阵指数最大误差_C": float(np.max(np.abs(rk45_fit - exact_fit)))})
    start = time.perf_counter(); rk4_fit = rk4_linear_solution(A_fit, b_fit, initial, times_s, num["rk4_step_s"]); rk4_time = time.perf_counter() - start
    solver_records.append({"求解器": "RK4", "设置": f"固定步长={num['rk4_step_s']} s", "运行时间_s": rk4_time, "相对矩阵指数最大误差_C": float(np.max(np.abs(rk4_fit - exact_fit)))})
    pd.DataFrame(solver_records).to_csv(RESULTS / "ode_solver_comparison.csv", index=False, encoding="utf-8-sig")

    # 新工况0—4h，每60s输出。
    pred_times = np.arange(0, pred["duration_s"] + pred["output_step_s"], pred["output_step_s"], dtype=float)
    A_pred, b_pred = charging_state_space(pred["flow_kg_s"], final_params[0], final_params[1], mass, prop["cp_j_kg_k"], pred["inlet_c"], prop["ambient_c"])
    prediction = exact_linear_solution(A_pred, b_pred, np.asarray(pred["initial_c"], dtype=float), pred_times)
    prediction_df = pd.DataFrame(prediction, columns=["T1_C", "T2_C", "T3_C", "T4_C", "T5_C"])
    prediction_df.insert(0, "time_s", pred_times.astype(int))
    prediction_df["bottom_outlet_C"] = prediction_df["T5_C"]
    prediction_df.to_csv(RESULTS / "prediction_60s.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    milestones = prediction_df[prediction_df["time_s"].isin([0, 3600, 7200, 10800, 14400])].copy()
    milestones.insert(1, "time_h", milestones["time_s"] / 3600.0)
    milestones.to_csv(RESULTS / "milestone_temperatures.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 能量守恒检查。
    derivatives = prediction @ A_pred.T + b_pred
    energy_residual = energy_balance_residual(prediction, derivatives, pred["flow_kg_s"], prop["cp_j_kg_k"], pred["inlet_c"], prop["ambient_c"], final_params[1], mass)
    energy_scale = np.maximum(1.0, np.abs(pred["flow_kg_s"] * prop["cp_j_kg_k"] * (pred["inlet_c"] - prediction[:, -1])))
    energy_max_w = float(np.max(np.abs(energy_residual)))
    energy_rel = float(np.max(np.abs(energy_residual) / energy_scale))

    # 局部无量纲灵敏度：参数±10%造成的观测预测变化。
    sensitivity_rows = []
    base_obs = final_fit[:, observed_idx]
    for p_idx, name in enumerate(["K", "UA"]):
        for change in [-0.10, 0.10]:
            changed = final_params.copy(); changed[p_idx] *= 1.0 + change
            delta = simulate(changed)[:, observed_idx] - base_obs
            sensitivity_rows.append({"参数": name, "相对变化": change, "预测RMSE变化_C": float(np.sqrt(np.mean(delta**2))), "预测最大变化_C": float(np.max(np.abs(delta))), "SSE相对最优值倍数": objective(changed) / final_metrics["SSE"]})
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(RESULTS / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")

    residual_df = []
    for ti, hour in enumerate(ident["times_h"]):
        for li, layer in enumerate(ident["observed_layers"]):
            residual_df.append({"time_h": hour, "layer": f"T{layer}", "observed_C": observations[ti, li], "fitted_C": final_fit[ti, observed_idx[li]], "residual_C": fit_residuals[ti, li]})
    pd.DataFrame(residual_df).to_csv(RESULTS / "fit_and_residuals.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

    parameter_output = {
        "K_W_per_K": float(final_params[0]),
        "UA_W_per_K": float(final_params[1]),
        "selected_method": str(best_row["算法"]),
        "parameter_bounds": {"K_W_per_K": [float(lower[0]), float(upper[0])], "UA_W_per_K": [float(lower[1]), float(upper[1])]},
        "normal_95pct_CI": {"K_W_per_K": normal_ci[0].tolist(), "UA_W_per_K": normal_ci[1].tolist()},
        "bootstrap_95pct_CI": {"K_W_per_K": boot_ci[0].tolist(), "UA_W_per_K": boot_ci[1].tolist()},
        "covariance_matrix": covariance.tolist(),
        "parameter_correlation": float(correlation),
        "scaled_jacobian_condition_number": condition_number,
        "fit_metrics": final_metrics,
        "measurement_error_limit_C": 0.6,
        "residuals_within_measurement_limit": bool(final_metrics["MaxAbsError_C"] <= 0.6),
        "mass_per_layer_kg": float(mass),
        "heat_capacity_per_layer_J_per_K": float(heat_capacity),
        "energy_balance_max_abs_W": energy_max_w,
        "energy_balance_max_relative": energy_rel,
        "data_source": "题目表1全部T1、T3、T5观测值（7时刻、21点）",
        "model": "五层充热集中参数模型；常系数增广矩阵指数求解",
        "created_by": "solve_q1.py",
    }
    (RESULTS / "identified_parameters.json").write_text(json.dumps(parameter_output, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图1：观测与拟合。
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, layer_idx, layer_name in zip(axes, observed_idx, ["T1", "T3", "T5"]):
        obs_col = list(observed_idx).index(layer_idx)
        ax.plot(ident["times_h"], observations[:, obs_col], "o", label="观测值", color="#D97706")
        ax.plot(ident["times_h"], final_fit[:, layer_idx], "-", label="拟合值", color="#2563A6")
        ax.set_title(f"{layer_name}观测—拟合")
        ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / ℃"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "fig1_observed_fitted.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图2：残差及±0.6℃带。
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for li, layer in enumerate(["T1", "T3", "T5"]):
        ax.plot(ident["times_h"], fit_residuals[:, li], "o-", label=layer)
    ax.axhspan(-0.6, 0.6, color="#16A34A", alpha=0.12, label="题设±0.6℃范围")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("时间 / h"); ax.set_ylabel("拟合残差 / ℃"); ax.set_title("参数辨识残差"); ax.grid(alpha=0.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "fig2_residuals.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图3：Bootstrap参数分布与相关性。
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(boot_df["K_W_per_K"], boot_df["UA_W_per_K"], s=12, alpha=0.28, color="#2563A6")
    ax.scatter([final_params[0]], [final_params[1]], marker="*", s=180, color="#DC2626", label="最优估计")
    ax.set_xlabel("K / (W/K)"); ax.set_ylabel("UA / (W/K)"); ax.set_title("Bootstrap参数联合分布"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "fig3_bootstrap_correlation.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图4：新工况五层温度。
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx in range(5):
        ax.plot(pred_times / 3600.0, prediction[:, idx], label=f"T{idx+1}")
    ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / ℃"); ax.set_title("新工况五层温度演化"); ax.grid(alpha=0.25); ax.legend(ncol=5)
    fig.tight_layout(); fig.savefig(FIGURES / "fig4_prediction_layers.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图5：相对灵敏度比较。
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [f"{r['参数']} {r['相对变化']:+.0%}" for r in sensitivity_rows]
    values = [r["预测RMSE变化_C"] for r in sensitivity_rows]
    ax.bar(labels, values, color=["#60A5FA", "#2563A6", "#FBBF24", "#D97706"])
    ax.set_ylabel("相对最优预测的RMSE / ℃"); ax.set_title("K与UA的±10%局部灵敏度"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURES / "fig5_sensitivity.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    LOGGER.info("最终参数 K=%.10f W/K, UA=%.10f W/K", final_params[0], final_params[1])
    LOGGER.info("拟合 SSE=%.10f RMSE=%.8f C MaxAE=%.8f C; ±0.6C内=%s", final_metrics["SSE"], final_metrics["RMSE_C"], final_metrics["MaxAbsError_C"], final_metrics["MaxAbsError_C"] <= 0.6)
    LOGGER.info("Bootstrap 95%% CI K=%s, UA=%s; corr=%.6f; cond=%.3f", boot_ci[0], boot_ci[1], correlation, condition_number)
    LOGGER.info("能量守恒最大绝对残差 %.6e W，相对残差 %.6e", energy_max_w, energy_rel)
    LOGGER.info("输出241时刻预测、5个里程碑和全部图表完成")


if __name__ == "__main__":
    main()
