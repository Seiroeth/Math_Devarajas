#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2025 A 题烟幕模型的公共几何、区间与严格复核函数。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

G = 9.8
SMOKE_RADIUS = 10.0
SMOKE_LIFETIME = 20.0
SMOKE_SINK_SPEED = 3.0

U0 = np.array([17800.0, 0.0, 1800.0])
UAV_POSITIONS = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}
MISSILE_SPEED = 300.0
MISSILE_POSITIONS = {
    "M1": (20000.0, 0.0, 2000.0),
    "M2": (19000.0, 600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}

TARGET_RADIUS = 7.0
TARGET_Y = 200.0
TARGET_LEVELS = (0.0, 10.0)


@dataclass(frozen=True)
class Missile:
    """来袭导弹；以 300 m/s 沿初始点至假目标原点作匀速直线运动。"""

    name: str
    m0: tuple[float, float, float]

    @property
    def initial_position(self) -> np.ndarray:
        return np.asarray(self.m0, dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        p0 = self.initial_position
        return -MISSILE_SPEED * p0 / np.linalg.norm(p0)

    @property
    def hit_time(self) -> float:
        return float(np.linalg.norm(self.initial_position) / MISSILE_SPEED)


MISSILES = {name: Missile(name, position) for name, position in MISSILE_POSITIONS.items()}
DEFAULT_MISSILE = MISSILES["M1"]

# 保留第 3、4 问使用的旧常量名，默认均指 M1。
M0 = DEFAULT_MISSILE.initial_position
MISSILE_V = DEFAULT_MISSILE.velocity
MISSILE_HIT_TIME = DEFAULT_MISSILE.hit_time
MAX_MISSILE_HIT_TIME = max(missile.hit_time for missile in MISSILES.values())


@dataclass(frozen=True)
class Smoke:
    """一枚烟幕弹策略；psi 使用弧度。"""

    psi: float
    speed: float
    td: float
    tau: float
    u0: tuple[float, float, float] = UAV_POSITIONS["FY1"]

    @property
    def initial_position(self) -> np.ndarray:
        return np.asarray(self.u0, dtype=float)

    @property
    def te(self) -> float:
        return self.td + self.tau

    @property
    def drop_point(self) -> np.ndarray:
        u0 = self.initial_position
        return np.array([
            u0[0] + self.speed * self.td * math.cos(self.psi),
            u0[1] + self.speed * self.td * math.sin(self.psi),
            u0[2],
        ])

    @property
    def explosion_point(self) -> np.ndarray:
        u0 = self.initial_position
        return np.array([
            u0[0] + self.speed * self.te * math.cos(self.psi),
            u0[1] + self.speed * self.te * math.sin(self.psi),
            u0[2] - 0.5 * G * self.tau**2,
        ])

    @property
    def valid(self) -> bool:
        return (
            70.0 <= self.speed <= 140.0
            and self.td >= 0.0
            and self.tau >= 0.0
            and self.explosion_point[2] >= 0.0
            and self.te < MAX_MISSILE_HIT_TIME
        )


def missile_position(
    t: float | np.ndarray,
    missile: Missile = DEFAULT_MISSILE,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return missile.initial_position + t_arr[..., None] * missile.velocity


def smoke_center(t: float | np.ndarray, smoke: Smoke) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    sink = np.zeros(t_arr.shape + (3,), dtype=float)
    sink[..., 2] = -SMOKE_SINK_SPEED * (t_arr - smoke.te)
    return smoke.explosion_point + sink


def target_ring_point(phi: float, h: float) -> np.ndarray:
    return np.array([
        TARGET_RADIUS * math.cos(phi),
        TARGET_Y + TARGET_RADIUS * math.sin(phi),
        h,
    ])


def point_to_segment_distance(c: np.ndarray, m: np.ndarray, x: np.ndarray) -> float:
    a = x - m
    b = c - m
    lam = float(np.dot(a, b) / np.dot(a, a))
    lam = min(1.0, max(0.0, lam))
    return float(np.linalg.norm(c - (m + lam * a)))


def max_ring_distance(
    t: float,
    smoke: Smoke,
    grid_n: int = 720,
    refine: bool = True,
    missile: Missile = DEFAULT_MISSILE,
) -> tuple[float, tuple[float, float]]:
    """上下两圆周最大视线段距离；refine=True 时连续角度精修。"""
    m = missile_position(t, missile=missile)
    c = smoke_center(t, smoke)
    phis = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)
    step = 2.0 * math.pi / grid_n
    best_d = -math.inf
    best_info = (0.0, 0.0)

    for h in TARGET_LEVELS:
        x = np.column_stack((
            TARGET_RADIUS * np.cos(phis),
            TARGET_Y + TARGET_RADIUS * np.sin(phis),
            np.full(grid_n, h),
        ))
        a = x - m
        b = c - m
        lam = np.clip((a @ b) / np.sum(a * a, axis=1), 0.0, 1.0)
        q = m + lam[:, None] * a
        distances = np.linalg.norm(c - q, axis=1)

        for idx in np.argsort(distances)[-6:]:
            if refine:
                center = phis[idx]

                def objective(p: float) -> float:
                    xp = target_ring_point(p % (2.0 * math.pi), h)
                    return -point_to_segment_distance(c, m, xp)

                result = minimize_scalar(
                    objective,
                    bounds=(center - step, center + step),
                    method="bounded",
                    options={"xatol": 1e-12},
                )
                distance = -float(result.fun)
                phi = float(result.x % (2.0 * math.pi))
            else:
                distance = float(distances[idx])
                phi = float(phis[idx])
            if distance > best_d:
                best_d = distance
                best_info = (h, phi)
    return best_d, best_info


def exact_margin(
    t: float,
    smoke: Smoke,
    grid_n: int = 720,
    missile: Missile = DEFAULT_MISSILE,
) -> float:
    return max_ring_distance(
        t, smoke, grid_n=grid_n, refine=True, missile=missile,
    )[0] - SMOKE_RADIUS


def grid_margins(
    ts: np.ndarray,
    smoke: Smoke,
    nphi: int = 72,
    missile: Missile = DEFAULT_MISSILE,
) -> np.ndarray:
    """批量离散圆周判据，供优化阶段使用。"""
    ts = np.asarray(ts, dtype=float)
    if ts.size == 0:
        return np.empty(0)
    phis = np.linspace(0.0, 2.0 * math.pi, nphi, endpoint=False)
    m = missile_position(ts, missile=missile)
    c = smoke_center(ts, smoke)
    b = c - m
    worst = np.full(ts.size, -math.inf)

    for h in TARGET_LEVELS:
        x = np.column_stack((
            TARGET_RADIUS * np.cos(phis),
            TARGET_Y + TARGET_RADIUS * np.sin(phis),
            np.full(nphi, h),
        ))
        a = x[None, :, :] - m[:, None, :]
        den = np.sum(a * a, axis=2)
        lam = np.clip(np.sum(a * b[:, None, :], axis=2) / den, 0.0, 1.0)
        q = m[:, None, :] + lam[:, :, None] * a
        distances = np.linalg.norm(c[:, None, :] - q, axis=2)
        worst = np.maximum(worst, np.max(distances, axis=1))
    return worst - SMOKE_RADIUS


def _intervals_from_samples(ts: np.ndarray, margins: np.ndarray) -> list[tuple[float, float]]:
    """用线性插值把布尔时间样本转换为近似区间。"""
    if ts.size == 0:
        return []
    inside = margins <= 0.0
    intervals: list[tuple[float, float]] = []
    start = float(ts[0]) if inside[0] else None

    for i in range(1, ts.size):
        if inside[i] == inside[i - 1]:
            continue
        f0, f1 = float(margins[i - 1]), float(margins[i])
        if abs(f1 - f0) < 1e-14:
            crossing = 0.5 * float(ts[i - 1] + ts[i])
        else:
            crossing = float(ts[i - 1] - f0 * (ts[i] - ts[i - 1]) / (f1 - f0))
        if inside[i]:
            start = crossing
        elif start is not None:
            intervals.append((start, crossing))
            start = None
    if start is not None:
        intervals.append((start, float(ts[-1])))
    return intervals


def fast_intervals(
    smoke: Smoke,
    dt: float = 0.08,
    nphi: int = 72,
    missile: Missile = DEFAULT_MISSILE,
) -> list[tuple[float, float]]:
    if not smoke.valid or smoke.te >= missile.hit_time:
        return []
    end = min(smoke.te + SMOKE_LIFETIME, missile.hit_time)
    ts = np.arange(smoke.te, end, dt)
    if ts.size == 0 or ts[-1] < end:
        ts = np.append(ts, end)
    return _intervals_from_samples(
        ts, grid_margins(ts, smoke, nphi=nphi, missile=missile),
    )


def exact_intervals(
    smoke: Smoke,
    scan_dt: float = 0.05,
    scan_nphi: int = 360,
    exact_nphi: int = 720,
    missile: Missile = DEFAULT_MISSILE,
) -> list[tuple[float, float]]:
    """粗扫定位后以连续角度极值和 Brent 根求解严格区间。"""
    if not smoke.valid or smoke.te >= missile.hit_time:
        return []
    end = min(smoke.te + SMOKE_LIFETIME, missile.hit_time)
    ts = np.arange(smoke.te, end, scan_dt)
    if ts.size == 0 or ts[-1] < end:
        ts = np.append(ts, end)
    margins = grid_margins(ts, smoke, nphi=scan_nphi, missile=missile)
    roots: list[float] = []

    def refined_roots(left: float, right: float) -> list[float]:
        """粗网格与连续角度极值不一致时，在扩展窗口内重新寻找精确括区。"""
        probe = np.linspace(left, right, 13)
        values = [
            exact_margin(float(t), smoke, grid_n=exact_nphi, missile=missile)
            for t in probe
        ]
        found: list[float] = []
        for j in range(len(probe) - 1):
            g0, g1 = values[j], values[j + 1]
            if abs(g0) <= 1e-12:
                found.append(float(probe[j]))
            elif g0 * g1 < 0.0:
                found.append(float(brentq(
                    lambda t: exact_margin(
                        t, smoke, grid_n=exact_nphi, missile=missile,
                    ),
                    float(probe[j]),
                    float(probe[j + 1]),
                    xtol=1e-10,
                )))
        return found

    for i in range(ts.size - 1):
        f0, f1 = float(margins[i]), float(margins[i + 1])
        if f0 == 0.0:
            roots.append(float(ts[i]))
        elif f0 * f1 < 0.0:
            left = float(ts[max(0, i - 1)])
            right = float(ts[min(ts.size - 1, i + 2)])
            roots.extend(refined_roots(left, right))

    roots = sorted(set(round(root, 10) for root in roots))
    cuts = [smoke.te, *roots, end]
    intervals: list[tuple[float, float]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a <= 1e-10:
            continue
        if exact_margin(
            0.5 * (a + b), smoke, grid_n=exact_nphi, missile=missile,
        ) <= 0.0:
            intervals.append((float(a), float(b)))
    return intervals


def merge_intervals(intervals: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged = [ordered[0]]
    for a, b in ordered[1:]:
        old_a, old_b = merged[-1]
        if a <= old_b + tol:
            merged[-1] = (old_a, max(old_b, b))
        else:
            merged.append((a, b))
    return merged


def interval_length(intervals: list[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in intervals))


def independent_union(
    smokes: list[Smoke],
    exact: bool = False,
    dt: float = 0.08,
    nphi: int = 72,
    missile: Missile = DEFAULT_MISSILE,
) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float]], float]:
    each = [
        exact_intervals(s, missile=missile)
        if exact
        else fast_intervals(s, dt=dt, nphi=nphi, missile=missile)
        for s in smokes
    ]
    merged = merge_intervals([interval_ for intervals in each for interval_ in intervals])
    return each, merged, interval_length(merged)


def sample_target_surface(
    theta_step_deg: float = 2.0,
    z_step: float = 1.0,
    radial_step: float = 1.0,
) -> np.ndarray:
    """圆柱侧面及上下底面的离散表面点，用于多烟幕 max-min 精检。"""
    phis = np.deg2rad(np.arange(0.0, 360.0, theta_step_deg))
    points: list[tuple[float, float, float]] = []
    for z in np.arange(0.0, 10.0 + 0.5 * z_step, z_step):
        points.extend((
            TARGET_RADIUS * math.cos(phi),
            TARGET_Y + TARGET_RADIUS * math.sin(phi),
            float(z),
        ) for phi in phis)
    for z in TARGET_LEVELS:
        for radius in np.arange(0.0, TARGET_RADIUS, radial_step):
            if radius == 0.0:
                points.append((0.0, TARGET_Y, z))
            else:
                points.extend((
                    radius * math.cos(phi),
                    TARGET_Y + radius * math.sin(phi),
                    z,
                ) for phi in phis)
    return np.unique(np.round(np.asarray(points), 10), axis=0)


def multi_smoke_margin(
    t: float,
    smokes: list[Smoke],
    target_points: np.ndarray,
    missile: Missile = DEFAULT_MISSILE,
) -> float:
    active = [
        s for s in smokes
        if s.te <= t <= min(s.te + SMOKE_LIFETIME, missile.hit_time)
    ]
    if not active:
        return math.inf
    m = missile_position(t, missile=missile)
    a = target_points - m
    den = np.sum(a * a, axis=1)
    best = np.full(target_points.shape[0], math.inf)
    for smoke in active:
        c = smoke_center(t, smoke)
        b = c - m
        lam = np.clip((a @ b) / den, 0.0, 1.0)
        q = m + lam[:, None] * a
        best = np.minimum(best, np.linalg.norm(c - q, axis=1))
    return float(np.max(best) - SMOKE_RADIUS)


def cooperative_intervals(
    smokes: list[Smoke],
    time_dt: float = 0.02,
    theta_step_deg: float = 2.0,
    z_step: float = 1.0,
    radial_step: float = 1.0,
    missile: Missile = DEFAULT_MISSILE,
) -> tuple[list[tuple[float, float]], int]:
    """完整表面采样下的多烟幕 max-min 联合遮蔽区间。"""
    target = sample_target_surface(theta_step_deg, z_step, radial_step)
    events = sorted(set(
        [s.te for s in smokes]
        + [min(s.te + SMOKE_LIFETIME, missile.hit_time) for s in smokes]
    ))
    intervals: list[tuple[float, float]] = []

    for left, right in zip(events[:-1], events[1:]):
        if right <= left:
            continue
        ts = np.arange(left, right, time_dt)
        if ts.size == 0 or ts[-1] < right:
            ts = np.append(ts, right)
        margins = np.array([
            multi_smoke_margin(float(t), smokes, target, missile=missile)
            for t in ts
        ])
        roots: list[float] = []
        for i in range(ts.size - 1):
            f0, f1 = float(margins[i]), float(margins[i + 1])
            if f0 == 0.0:
                roots.append(float(ts[i]))
            elif np.isfinite(f0) and np.isfinite(f1) and f0 * f1 < 0.0:
                roots.append(float(brentq(
                    lambda t: multi_smoke_margin(
                        t, smokes, target, missile=missile,
                    ),
                    float(ts[i]),
                    float(ts[i + 1]),
                    xtol=1e-9,
                )))
        cuts = [left, *sorted(set(round(root, 9) for root in roots)), right]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= 1e-10:
                continue
            if multi_smoke_margin(
                0.5 * (a + b), smokes, target, missile=missile,
            ) <= 0.0:
                intervals.append((float(a), float(b)))
    return merge_intervals(intervals), int(target.shape[0])
