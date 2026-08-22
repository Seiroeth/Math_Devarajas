#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 高教社杯 A题 - 问题2
采用“完整圆柱目标 + 上下两圆周严格判据”。

本文件提供两种用途：
1) 默认：高精度复核当前已经搜索到的一组近优候选；
2) --optimize：用差分进化重新搜索，再对最佳候选做严格复算。

当前已知稳定近优候选（g=9.8）：
speed   ≈ 70.0 m/s
heading ≈ 176.643 deg
td      ≈ 0 s
tau     ≈ 2.49705 s
严格复算遮蔽时长 ≈ 4.542879 s

注意：
- 这是数值近优解，不是解析证明的全局最优。
- --optimize 的全局搜索阶段使用较粗时间/圆周离散以控制计算量，
  最后会调用严格判据复算。
"""

import argparse
import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq, differential_evolution

G = 9.8
SMOKE_RADIUS = 10.0
SMOKE_LIFETIME = 20.0
SMOKE_SINK_SPEED = 3.0

M0 = np.array([20000.0, 0.0, 2000.0])
U0 = np.array([17800.0, 0.0, 1800.0])

MISSILE_SPEED = 300.0
MISSILE_DIR = -M0 / np.linalg.norm(M0)
MISSILE_V = MISSILE_SPEED * MISSILE_DIR
MISSILE_HIT_TIME = np.linalg.norm(M0) / MISSILE_SPEED

TARGET_RADIUS = 7.0
TARGET_Y = 200.0
TARGET_LEVELS = (0.0, 10.0)

# 当前已经得到的近优候选，用于默认高精度复核
KNOWN = {
    "psi_deg": 176.643,
    "speed": 70.0,
    "td": 0.0,
    "tau": 2.49705,
}


def missile_position(t: float) -> np.ndarray:
    return M0 + MISSILE_V * t


def target_ring_point(phi: float, h: float) -> np.ndarray:
    return np.array([
        TARGET_RADIUS * math.cos(phi),
        TARGET_Y + TARGET_RADIUS * math.sin(phi),
        h
    ], dtype=float)


def point_to_segment_distance(C: np.ndarray, M: np.ndarray, X: np.ndarray) -> float:
    a = X - M
    b = C - M
    lam = float(np.dot(a, b) / np.dot(a, a))
    lam = min(1.0, max(0.0, lam))
    Q = M + lam * a
    return float(np.linalg.norm(C - Q))


def max_ring_distance(t: float, C: np.ndarray,
                      grid_n: int = 720,
                      refine: bool = True) -> tuple[float, tuple[float, float]]:
    M = missile_position(t)
    phis = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)
    step = 2.0 * math.pi / grid_n

    best_d = -1.0
    best_info = (0.0, 0.0)

    for h in TARGET_LEVELS:
        X = np.column_stack([
            TARGET_RADIUS * np.cos(phis),
            TARGET_Y + TARGET_RADIUS * np.sin(phis),
            np.full(grid_n, h)
        ])

        a = X - M
        b = C - M
        den = np.sum(a * a, axis=1)
        lam = (a @ b) / den
        lam = np.clip(lam, 0.0, 1.0)
        Q = M + lam[:, None] * a
        d = np.linalg.norm(C - Q, axis=1)

        candidate_ids = np.argsort(d)[-6:]

        for idx in candidate_ids:
            if refine:
                center = phis[idx]

                def objective(p: float) -> float:
                    Xp = target_ring_point(p % (2.0 * math.pi), h)
                    return -point_to_segment_distance(C, M, Xp)

                res = minimize_scalar(
                    objective,
                    bounds=(center - step, center + step),
                    method="bounded",
                    options={"xatol": 1e-12}
                )
                cur_d = -float(res.fun)
                cur_phi = float(res.x % (2.0 * math.pi))
            else:
                cur_d = float(d[idx])
                cur_phi = float(phis[idx])

            if cur_d > best_d:
                best_d = cur_d
                best_info = (h, cur_phi)

    return best_d, best_info


def strategy_geometry(psi: float, speed: float, td: float, tau: float):
    te = td + tau

    P_drop = np.array([
        U0[0] + speed * td * math.cos(psi),
        U0[1] + speed * td * math.sin(psi),
        U0[2]
    ])

    P_explode = np.array([
        U0[0] + speed * te * math.cos(psi),
        U0[1] + speed * te * math.sin(psi),
        U0[2] - 0.5 * G * tau * tau
    ])

    return te, P_drop, P_explode


def smoke_center(t: float, te: float, P_explode: np.ndarray) -> np.ndarray:
    return P_explode + np.array([0.0, 0.0, -SMOKE_SINK_SPEED * (t - te)])


def exact_margin(t: float, psi: float, speed: float, td: float, tau: float,
                 grid_n: int = 720) -> float:
    te, _, P_explode = strategy_geometry(psi, speed, td, tau)
    C = smoke_center(t, te, P_explode)
    D, _ = max_ring_distance(t, C, grid_n=grid_n, refine=True)
    return D - SMOKE_RADIUS


def exact_intervals(psi: float, speed: float, td: float, tau: float,
                    scan_dt: float = 0.05) -> list[tuple[float, float]]:
    te, _, P_explode = strategy_geometry(psi, speed, td, tau)

    if td < 0.0 or tau < 0.0:
        return []
    if P_explode[2] < 0.0:
        return []
    if te >= MISSILE_HIT_TIME:
        return []

    t_end = min(te + SMOKE_LIFETIME, MISSILE_HIT_TIME)
    ts = np.arange(te, t_end + 0.5 * scan_dt, scan_dt)
    if ts[-1] < t_end:
        ts = np.append(ts, t_end)

    # 粗扫只用网格角度
    vals = []
    for t in ts:
        C = smoke_center(t, te, P_explode)
        D, _ = max_ring_distance(t, C, grid_n=360, refine=False)
        vals.append(D - SMOKE_RADIUS)
    vals = np.asarray(vals)

    roots = []
    for i in range(len(ts) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if f1 == 0.0:
            roots.append(float(ts[i]))
        elif f1 * f2 < 0.0:
            r = brentq(
                lambda t: exact_margin(t, psi, speed, td, tau, grid_n=720),
                float(ts[i]), float(ts[i + 1]),
                xtol=1e-10
            )
            roots.append(float(r))

    roots = sorted(set(round(r, 10) for r in roots))
    cuts = [te] + roots + [t_end]
    intervals = []

    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-10:
            continue
        mid = 0.5 * (a + b)
        if exact_margin(mid, psi, speed, td, tau, grid_n=720) <= 0.0:
            intervals.append((float(a), float(b)))

    return intervals


def interval_total(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


# ---------- 用于差分进化的快速近似目标 ----------

def fast_duration(x, dt: float = 0.10, nphi: int = 72) -> float:
    """
    x = [psi_deg, speed, td, tau]
    用离散时间 + 离散圆周得到快速近似遮蔽时长。
    仅用于全局搜索；最终答案必须 exact_intervals() 复算。
    """
    psi_deg, speed, td, tau = map(float, x)
    psi = math.radians(psi_deg)

    if not (70.0 <= speed <= 140.0):
        return 0.0
    if td < 0.0 or tau < 0.0:
        return 0.0

    te, _, P_explode = strategy_geometry(psi, speed, td, tau)

    if P_explode[2] < 0.0 or te >= MISSILE_HIT_TIME:
        return 0.0

    t_end = min(te + SMOKE_LIFETIME, MISSILE_HIT_TIME)
    ts = np.arange(te, t_end + 1e-12, dt)
    if len(ts) == 0:
        return 0.0

    phis = np.linspace(0.0, 2.0 * math.pi, nphi, endpoint=False)
    cos_p = np.cos(phis)
    sin_p = np.sin(phis)

    inside = np.zeros(len(ts), dtype=bool)

    for it, t in enumerate(ts):
        M = missile_position(t)
        C = smoke_center(t, te, P_explode)

        worst = -1.0
        for h in TARGET_LEVELS:
            X = np.column_stack([
                TARGET_RADIUS * cos_p,
                TARGET_Y + TARGET_RADIUS * sin_p,
                np.full(nphi, h)
            ])

            a = X - M
            b = C - M
            den = np.sum(a * a, axis=1)
            lam = (a @ b) / den
            lam = np.clip(lam, 0.0, 1.0)
            Q = M + lam[:, None] * a
            d = np.linalg.norm(C - Q, axis=1)
            worst = max(worst, float(np.max(d)))

        inside[it] = (worst <= SMOKE_RADIUS)

    # 用布尔采样近似 measure
    return float(np.sum(inside) * dt)


def objective_fast(x):
    return -fast_duration(x)


def optimize():
    """
    默认使用“知情但仍较宽”的搜索范围。
    若希望真正从 0~360° 全航向搜索，可把 psi bounds 改为 (0,360)，
    同时相应增大 maxiter/popsize。
    """
    tau_max = math.sqrt(2.0 * U0[2] / G)

    bounds = [
        (150.0, 210.0),   # psi_deg：基于几何分析的宽范围
        (70.0, 140.0),    # speed
        (0.0, 10.0),      # td
        (0.05, min(10.0, tau_max)),  # tau
    ]

    res = differential_evolution(
        objective_fast,
        bounds=bounds,
        seed=2025,
        popsize=14,
        maxiter=100,
        tol=1e-7,
        polish=False,
        updating="immediate",
        workers=1,
    )

    return res.x, -float(res.fun)


def report(psi_deg: float, speed: float, td: float, tau: float):
    psi = math.radians(psi_deg)
    te, P_drop, P_explode = strategy_geometry(psi, speed, td, tau)
    intervals = exact_intervals(psi, speed, td, tau)
    T = interval_total(intervals)

    print("=== Q2 严格圆柱判据 ===")
    print(f"speed           : {speed:.9f} m/s")
    print(f"heading         : {psi_deg:.9f} deg")
    print(f"drop time       : {td:.9f} s")
    print(f"fuse delay      : {tau:.9f} s")
    print(f"explode time    : {te:.9f} s")
    print("drop point      :", np.array2string(P_drop, precision=6))
    print("explode point   :", np.array2string(P_explode, precision=6))
    print("shield intervals:")
    for a, b in intervals:
        print(f"  [{a:.10f}, {b:.10f}]  duration={b-a:.10f} s")
    print(f"TOTAL SHIELDING : {T:.10f} s")
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="重新运行差分进化搜索；否则复核当前已知近优候选"
    )
    args = parser.parse_args()

    if not args.optimize:
        report(
            KNOWN["psi_deg"],
            KNOWN["speed"],
            KNOWN["td"],
            KNOWN["tau"]
        )
        return

    x, approx_T = optimize()
    psi_deg, speed, td, tau = map(float, x)

    print("=== 差分进化粗搜索结果 ===")
    print("x =", x)
    print(f"fast approx duration = {approx_T:.6f} s")
    print()
    print("=== 对搜索结果做严格复算 ===")
    report(psi_deg, speed, td, tau)

    print()
    print("=== 当前已知候选严格复算（用于交叉检查） ===")
    report(
        KNOWN["psi_deg"],
        KNOWN["speed"],
        KNOWN["td"],
        KNOWN["tau"]
    )


if __name__ == "__main__":
    main()
