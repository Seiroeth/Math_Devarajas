#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 高教社杯 A题 - 问题1
采用“完整圆柱目标 + 上下两圆周严格判据”。

固定条件：
FY1 初始 (17800,0,1800)，速度 120 m/s，朝假目标方向等高度飞行；
1.5 s 投弹，3.6 s 后起爆；
M1 初始 (20000,0,2000)，300 m/s 直指原点；
烟幕球有效半径 10 m，起爆后 20 s 有效，云心以 3 m/s 下沉。

预期结果（g=9.8）：
投弹点 (17620, 0, 1800)
起爆点 (17188, 0, 1736.496)
有效遮蔽区间约 [8.05644549, 9.44808816] s
有效遮蔽时长约 1.39164267 s
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq

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


def missile_position(t: float) -> np.ndarray:
    return M0 + MISSILE_V * t


def target_ring_point(phi: float, h: float) -> np.ndarray:
    return np.array([
        TARGET_RADIUS * math.cos(phi),
        TARGET_Y + TARGET_RADIUS * math.sin(phi),
        h
    ], dtype=float)


def point_to_segment_distance(C: np.ndarray, M: np.ndarray, X: np.ndarray) -> float:
    """点 C 到线段 MX 的最短距离。"""
    a = X - M
    b = C - M
    lam = float(np.dot(a, b) / np.dot(a, a))
    lam = min(1.0, max(0.0, lam))
    Q = M + lam * a
    return float(np.linalg.norm(C - Q))


def max_ring_distance(t: float, C: np.ndarray,
                      grid_n: int = 720,
                      refine: bool = True) -> tuple[float, tuple[float, float]]:
    """
    返回上下两个目标圆周中，烟幕中心到“导弹-目标点视线段”的最大距离。
    返回：(最大距离, (最不利高度h, 最不利角phi))
    """
    M = missile_position(t)
    phis = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)

    best_d = -1.0
    best_info = (0.0, 0.0)
    step = 2.0 * math.pi / grid_n

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

        # 粗网格最大值附近取多个候选，防止漏掉另一局部峰。
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


def build_q1_strategy():
    speed = 120.0
    psi = math.pi              # 等高度朝假目标：负 x 方向
    td = 1.5
    tau = 3.6
    te = td + tau

    velocity_u = np.array([
        speed * math.cos(psi),
        speed * math.sin(psi),
        0.0
    ])

    P_drop = U0 + velocity_u * td

    P_explode = np.array([
        U0[0] + speed * te * math.cos(psi),
        U0[1] + speed * te * math.sin(psi),
        U0[2] - 0.5 * G * tau * tau
    ])

    return speed, psi, td, tau, te, P_drop, P_explode


def smoke_center(t: float, te: float, P_explode: np.ndarray) -> np.ndarray:
    return P_explode + np.array([0.0, 0.0, -SMOKE_SINK_SPEED * (t - te)])


def margin(t: float, te: float, P_explode: np.ndarray,
           grid_n: int = 720, refine: bool = True) -> float:
    C = smoke_center(t, te, P_explode)
    D, _ = max_ring_distance(t, C, grid_n=grid_n, refine=refine)
    return D - SMOKE_RADIUS


def find_shield_intervals(te: float, P_explode: np.ndarray,
                          scan_dt: float = 0.05) -> list[tuple[float, float]]:
    """
    扫描 margin(t)=D(t)-10，再用 Brent 法精修边界。
    """
    t_end = min(te + SMOKE_LIFETIME, MISSILE_HIT_TIME)
    ts = np.arange(te, t_end + 0.5 * scan_dt, scan_dt)
    if ts[-1] < t_end:
        ts = np.append(ts, t_end)

    vals = np.array([
        margin(t, te, P_explode, grid_n=360, refine=False)
        for t in ts
    ])

    roots = []
    for i in range(len(ts) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if f1 == 0.0:
            roots.append(float(ts[i]))
        elif f1 * f2 < 0.0:
            r = brentq(
                lambda t: margin(t, te, P_explode, grid_n=720, refine=True),
                float(ts[i]), float(ts[i + 1]),
                xtol=1e-11
            )
            roots.append(float(r))

    # 去重
    roots = sorted(set(round(r, 10) for r in roots))

    # 用根和端点切分区间，再用中点判断是否遮蔽
    cuts = [te] + roots + [t_end]
    intervals = []

    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-10:
            continue
        mid = 0.5 * (a + b)
        if margin(mid, te, P_explode, grid_n=720, refine=True) <= 0.0:
            intervals.append((float(a), float(b)))

    return intervals


def total_length(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


def main():
    speed, psi, td, tau, te, P_drop, P_explode = build_q1_strategy()
    intervals = find_shield_intervals(te, P_explode)
    T = total_length(intervals)

    print("=== Q1 严格圆柱判据 ===")
    print(f"M1 到假目标时间: {MISSILE_HIT_TIME:.9f} s")
    print(f"FY1 speed       : {speed:.6f} m/s")
    print(f"FY1 heading     : {math.degrees(psi):.6f} deg")
    print(f"drop time       : {td:.9f} s")
    print(f"fuse delay      : {tau:.9f} s")
    print(f"explode time    : {te:.9f} s")
    print("drop point      :", np.array2string(P_drop, precision=6))
    print("explode point   :", np.array2string(P_explode, precision=6))
    print("shield intervals:")
    for a, b in intervals:
        print(f"  [{a:.10f}, {b:.10f}]  duration={b-a:.10f} s")
    print(f"TOTAL SHIELDING : {T:.10f} s")


if __name__ == "__main__":
    main()
