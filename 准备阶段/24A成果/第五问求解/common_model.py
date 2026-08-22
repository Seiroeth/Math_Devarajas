from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar


N_BOARDS = 223
N_HANDLES = N_BOARDS + 1
HEAD_HANDLE_DISTANCE = 3.41 - 2 * 0.275
BODY_HANDLE_DISTANCE = 2.20 - 2 * 0.275
BOARD_WIDTH = 0.30
END_OVERHANG = 0.275
INITIAL_THETA = 32 * math.pi
PAIR_I, PAIR_J = np.triu_indices(N_BOARDS, k=2)


def handle_distance(i: int) -> float:
    """Distance between handle i-1 and i, for i=1,...,223."""
    return HEAD_HANDLE_DISTANCE if i == 1 else BODY_HANDLE_DISTANCE


def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def spiral_b(pitch: float) -> float:
    return pitch / (2 * math.pi)


def spiral_point(theta: float, pitch: float) -> np.ndarray:
    b = spiral_b(pitch)
    return np.array([b * theta * math.cos(theta), b * theta * math.sin(theta)])


def spiral_derivative(theta: float, pitch: float) -> np.ndarray:
    b = spiral_b(pitch)
    return b * np.array(
        [math.cos(theta) - theta * math.sin(theta),
         math.sin(theta) + theta * math.cos(theta)]
    )


def spiral_forward_tangent(theta: float, pitch: float) -> np.ndarray:
    """Forward tangent for clockwise inward travel (theta decreases)."""
    d = spiral_derivative(theta, pitch)
    return -d / np.linalg.norm(d)


def spiral_arc_primitive(theta: float, pitch: float) -> float:
    b = spiral_b(pitch)
    return 0.5 * b * (
        theta * math.sqrt(1 + theta * theta) + math.asinh(theta)
    )


def invert_spiral_arc_from(theta_ref: float, distance: float, pitch: float) -> float:
    """Find theta >= theta_ref whose outward arc distance is distance."""
    if distance <= 0:
        return theta_ref
    b = spiral_b(pitch)
    target = spiral_arc_primitive(theta_ref, pitch) + distance
    theta = math.sqrt(max(theta_ref * theta_ref + 2 * distance / b, theta_ref**2))
    for _ in range(10):
        value = spiral_arc_primitive(theta, pitch) - target
        derivative = b * math.sqrt(1 + theta * theta)
        step = value / derivative
        theta -= step
        if abs(step) < 1e-13:
            break
    return theta


def head_theta_at_time(t: float, pitch: float = 0.55, speed: float = 1.0) -> float:
    target = spiral_arc_primitive(INITIAL_THETA, pitch) - speed * t
    if target <= 0:
        raise ValueError("Head has reached the spiral origin.")
    b = spiral_b(pitch)
    theta = math.sqrt(max(INITIAL_THETA**2 - 2 * speed * t / b, 1e-12))
    for _ in range(12):
        f = spiral_arc_primitive(theta, pitch) - target
        df = b * math.sqrt(1 + theta * theta)
        step = f / df
        theta -= step
        if abs(step) < 1e-13:
            break
    return theta


def next_spiral_theta(theta_prev: float, distance: float, pitch: float) -> float:
    prev = spiral_point(theta_prev, pitch)

    def equation(theta: float) -> float:
        delta = spiral_point(theta, pitch) - prev
        return float(delta @ delta - distance * distance)

    b = spiral_b(pitch)
    estimate = distance / (b * math.sqrt(1 + theta_prev * theta_prev))
    hi = theta_prev + max(estimate * 1.05, 1e-4)
    while equation(hi) <= 0:
        hi = theta_prev + (hi - theta_prev) * 1.25
        if hi - theta_prev > math.pi:
            raise RuntimeError("Failed to bracket the nearest spiral chord root.")
    return brentq(equation, theta_prev, hi, xtol=1e-13, rtol=1e-14)


def build_spiral_chain(theta_head: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.empty(N_HANDLES)
    points = np.empty((N_HANDLES, 2))
    thetas[0] = theta_head
    points[0] = spiral_point(theta_head, pitch)
    for i in range(1, N_HANDLES):
        thetas[i] = next_spiral_theta(thetas[i - 1], handle_distance(i), pitch)
        points[i] = spiral_point(thetas[i], pitch)
    return thetas, points


def velocities_from_tangents(
    points: np.ndarray, tangents: np.ndarray, head_speed: float = 1.0
) -> np.ndarray:
    speeds = np.empty(N_HANDLES)
    speeds[0] = head_speed
    for i in range(1, N_HANDLES):
        link = points[i] - points[i - 1]
        link /= np.linalg.norm(link)
        numerator = float(link @ tangents[i - 1])
        denominator = float(link @ tangents[i])
        if abs(denominator) < 1e-12:
            raise FloatingPointError("Velocity transfer denominator is too small.")
        speeds[i] = speeds[i - 1] * numerator / denominator
    return speeds


def spiral_state(theta_head: float, pitch: float, head_speed: float = 1.0):
    thetas, points = build_spiral_chain(theta_head, pitch)
    tangents = np.array([spiral_forward_tangent(x, pitch) for x in thetas])
    speeds = velocities_from_tangents(points, tangents, head_speed)
    return thetas, points, tangents, speeds


@dataclass
class Rectangle:
    center: np.ndarray
    axis_long: np.ndarray
    axis_wide: np.ndarray
    half_length: float
    half_width: float
    radius: float


def chain_rectangles(points: np.ndarray) -> list[Rectangle]:
    rectangles: list[Rectangle] = []
    for i in range(1, N_HANDLES):
        d = handle_distance(i)
        axis_long = (points[i] - points[i - 1]) / d
        axis_wide = rot90(axis_long)
        half_length = 0.5 * (d + 2 * END_OVERHANG)
        half_width = 0.5 * BOARD_WIDTH
        rectangles.append(
            Rectangle(
                center=0.5 * (points[i] + points[i - 1]),
                axis_long=axis_long,
                axis_wide=axis_wide,
                half_length=half_length,
                half_width=half_width,
                radius=math.hypot(half_length, half_width),
            )
        )
    return rectangles


def rectangle_sat_gap(a: Rectangle, b: Rectangle) -> float:
    """Positive if separated, zero at contact, negative if overlapping."""
    delta = b.center - a.center
    gaps = []
    for axis in (a.axis_long, a.axis_wide, b.axis_long, b.axis_wide):
        projected_centers = abs(float(delta @ axis))
        radius_a = (
            a.half_length * abs(float(a.axis_long @ axis))
            + a.half_width * abs(float(a.axis_wide @ axis))
        )
        radius_b = (
            b.half_length * abs(float(b.axis_long @ axis))
            + b.half_width * abs(float(b.axis_wide @ axis))
        )
        gaps.append(projected_centers - radius_a - radius_b)
    return max(gaps)


def minimum_clearance(points: np.ndarray, broad_margin: float = 0.5) -> tuple[float, tuple[int, int] | None]:
    links = points[1:] - points[:-1]
    distances = np.r_[HEAD_HANDLE_DISTANCE, np.full(N_BOARDS - 1, BODY_HANDLE_DISTANCE)]
    axis_long = links / distances[:, None]
    axis_wide = np.column_stack((-axis_long[:, 1], axis_long[:, 0]))
    centers = 0.5 * (points[1:] + points[:-1])
    half_length = 0.5 * (distances + 2 * END_OVERHANG)
    half_width = np.full(N_BOARDS, 0.5 * BOARD_WIDTH)
    radii = np.hypot(half_length, half_width)

    i_all, j_all = PAIR_I, PAIR_J
    delta_all = centers[j_all] - centers[i_all]
    center_distance = np.linalg.norm(delta_all, axis=1)
    mask = center_distance <= radii[i_all] + radii[j_all] + broad_margin
    if not np.any(mask):
        return broad_margin, None

    ii, jj = i_all[mask], j_all[mask]
    delta = delta_all[mask]
    axes = (
        axis_long[ii], axis_wide[ii], axis_long[jj], axis_wide[jj]
    )
    all_gaps = []
    for axis in axes:
        projected_centers = np.abs(np.einsum("ij,ij->i", delta, axis))
        radius_i = (
            half_length[ii] * np.abs(np.einsum("ij,ij->i", axis_long[ii], axis))
            + half_width[ii] * np.abs(np.einsum("ij,ij->i", axis_wide[ii], axis))
        )
        radius_j = (
            half_length[jj] * np.abs(np.einsum("ij,ij->i", axis_long[jj], axis))
            + half_width[jj] * np.abs(np.einsum("ij,ij->i", axis_wide[jj], axis))
        )
        all_gaps.append(projected_centers - radius_i - radius_j)
    pair_gaps = np.max(np.vstack(all_gaps), axis=0)
    k = int(np.argmin(pair_gaps))
    best = float(pair_gaps[k])
    if best >= broad_margin:
        return broad_margin, None
    return best, (int(ii[k] + 1), int(jj[k] + 1))


def collision_state_at_time(t: float, pitch: float = 0.55):
    theta = head_theta_at_time(t, pitch)
    _, points = build_spiral_chain(theta, pitch)
    clearance, pair = minimum_clearance(points)
    return clearance, pair, theta, points


def find_terminal_time(scan_step: float = 1.0) -> dict:
    t_left = 0.0
    c_left, _, _, _ = collision_state_at_time(t_left)
    if c_left <= 0:
        raise RuntimeError("The initial configuration is already colliding.")
    t_right = scan_step
    while t_right <= 1000:
        c_right, _, _, _ = collision_state_at_time(t_right)
        if c_right <= 0:
            break
        t_left, c_left = t_right, c_right
        t_right += scan_step
    else:
        raise RuntimeError("No collision was found before 1000 seconds.")

    def event(t: float) -> float:
        return collision_state_at_time(t)[0]

    root = brentq(event, t_left, t_right, xtol=1e-10, rtol=1e-12)
    clearance, pair, theta, points = collision_state_at_time(root)
    thetas, points, tangents, speeds = spiral_state(theta, 0.55, 1.0)
    return {
        "time": root,
        "clearance": clearance,
        "pair": pair,
        "theta_head": theta,
        "points": points,
        "speeds": speeds,
    }


def boundary_clearance_for_pitch(pitch: float):
    theta_boundary = 4.5 / spiral_b(pitch)
    if theta_boundary > INITIAL_THETA:
        return -math.inf, None, theta_boundary, None
    _, points = build_spiral_chain(theta_boundary, pitch)
    clearance, pair = minimum_clearance(points)
    return clearance, pair, theta_boundary, points


def path_clearance_for_pitch(pitch: float, grid_size: int = 31) -> dict:
    """Minimum clearance over the whole inward trip to the turn boundary."""
    theta_boundary = 4.5 / spiral_b(pitch)
    if theta_boundary > INITIAL_THETA:
        return {
            "clearance": -math.inf,
            "theta": theta_boundary,
            "pair": None,
        }

    cache: dict[float, tuple[float, tuple[int, int] | None]] = {}

    def evaluate(theta: float) -> float:
        key = round(float(theta), 12)
        if key not in cache:
            _, points = build_spiral_chain(float(theta), pitch)
            cache[key] = minimum_clearance(points)
        return cache[key][0]

    grid = np.linspace(theta_boundary, INITIAL_THETA, grid_size)
    values = np.array([evaluate(float(theta)) for theta in grid])
    candidates = set(np.argsort(values)[: min(6, grid_size)].tolist())
    for i in range(1, grid_size - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            candidates.add(i)

    best_value = float(values.min())
    best_theta = float(grid[int(np.argmin(values))])
    for i in candidates:
        if i == 0 or i == grid_size - 1:
            continue
        result = minimize_scalar(
            evaluate,
            bounds=(float(grid[i - 1]), float(grid[i + 1])),
            method="bounded",
            options={"xatol": 2e-7},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_theta = float(result.x)

    evaluate(best_theta)
    pair = cache[round(best_theta, 12)][1]
    return {
        "clearance": best_value,
        "theta": best_theta,
        "pair": pair,
        "evaluations": len(cache),
    }


def find_minimum_pitch() -> dict:
    low = 4.5 / 16 + 1e-6
    high = 0.80
    high_state = path_clearance_for_pitch(high)
    if high_state["clearance"] <= 0:
        raise RuntimeError("The upper pitch bound is not feasible.")
    for _ in range(30):
        mid = 0.5 * (low + high)
        state = path_clearance_for_pitch(mid)
        if state["clearance"] >= 0:
            high = mid
        else:
            low = mid
    pitch = high
    clearance, pair, theta_boundary, points = boundary_clearance_for_pitch(pitch)
    path_state = path_clearance_for_pitch(pitch, grid_size=81)
    return {
        "pitch": pitch,
        "boundary_clearance": clearance,
        "boundary_pair": pair,
        "theta_boundary": theta_boundary,
        "path_min_clearance": path_state["clearance"],
        "path_min_theta": path_state["theta"],
        "path_min_pair": path_state["pair"],
        "points": points,
    }


def oriented_angle(v0: np.ndarray, v1: np.ndarray, orientation: float) -> float:
    angle = math.atan2(cross2(v0, v1), float(v0 @ v1))
    if orientation > 0:
        return angle if angle >= 0 else angle + 2 * math.pi
    return -angle if angle <= 0 else 2 * math.pi - angle


@dataclass
class TurnPath:
    pitch: float = 1.7
    boundary_radius: float = 4.5
    radius_ratio: float = 2.0

    def __post_init__(self):
        self.theta_a = self.boundary_radius / spiral_b(self.pitch)
        self.a = spiral_point(self.theta_a, self.pitch)
        self.e = -self.a
        self.tangent = spiral_forward_tangent(self.theta_a, self.pitch)
        n_left = rot90(self.tangent)
        d = self.e - self.a
        self.normal = n_left if float(d @ n_left) > 0 else -n_left
        self.radius_sum = float(d @ d) / (2 * float(d @ self.normal))
        self.r1 = self.radius_sum * self.radius_ratio / (1 + self.radius_ratio)
        self.r2 = self.radius_sum / (1 + self.radius_ratio)
        self.o1 = self.a + self.r1 * self.normal
        self.o2 = self.e - self.r2 * self.normal
        center_vector = self.o2 - self.o1
        self.center_direction = center_vector / np.linalg.norm(center_vector)
        self.c = self.o1 + self.r1 * self.center_direction

        radial_a = (self.a - self.o1) / self.r1
        radial_c1 = (self.c - self.o1) / self.r1
        radial_c2 = (self.c - self.o2) / self.r2
        radial_e = (self.e - self.o2) / self.r2
        self.q1 = 1.0 if float(rot90(radial_a) @ self.tangent) > 0 else -1.0
        self.q2 = 1.0 if float(rot90(radial_e) @ self.tangent) > 0 else -1.0
        self.angle1 = oriented_angle(radial_a, radial_c1, self.q1)
        self.angle2 = oriented_angle(radial_c2, radial_e, self.q2)
        self.length1 = self.r1 * self.angle1
        self.length2 = self.r2 * self.angle2
        self.turn_length = self.length1 + self.length2
        self.radial_a = radial_a
        self.radial_c2 = radial_c2

    @staticmethod
    def rotate(v: np.ndarray, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def point_tangent(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        if s <= 0:
            theta = invert_spiral_arc_from(self.theta_a, -s, self.pitch)
            return spiral_point(theta, self.pitch), spiral_forward_tangent(theta, self.pitch)
        if s <= self.length1:
            radial = self.rotate(self.radial_a, self.q1 * s / self.r1)
            point = self.o1 + self.r1 * radial
            tangent = self.q1 * rot90(radial)
            return point, tangent
        if s <= self.turn_length:
            local = s - self.length1
            radial = self.rotate(self.radial_c2, self.q2 * local / self.r2)
            point = self.o2 + self.r2 * radial
            tangent = self.q2 * rot90(radial)
            return point, tangent
        theta = invert_spiral_arc_from(
            self.theta_a, s - self.turn_length, self.pitch
        )
        return -spiral_point(theta, self.pitch), spiral_forward_tangent(theta, self.pitch)

    def geometry_report(self) -> dict:
        samples = np.linspace(0, self.turn_length, 1001)
        radii = [np.linalg.norm(self.point_tangent(float(s))[0]) for s in samples]
        return {
            "theta_a": self.theta_a,
            "A": self.a,
            "E": self.e,
            "R1": self.r1,
            "R2": self.r2,
            "angle1": self.angle1,
            "angle2": self.angle2,
            "length1": self.length1,
            "length2": self.length2,
            "turn_length": self.turn_length,
            "max_radius_on_turn": max(radii),
            "junction_position_error": float(
                np.linalg.norm(self.point_tangent(self.length1)[0] - self.c)
            ),
        }


def next_path_coordinate(path: TurnPath, s_prev: float, distance: float) -> float:
    p_prev, _ = path.point_tangent(s_prev)

    def equation(s: float) -> float:
        p, _ = path.point_tangent(s)
        delta = p - p_prev
        return float(delta @ delta - distance * distance)

    s_hi = s_prev
    f_hi = -distance * distance
    step = min(0.05, distance / 30)
    s_lo = s_hi - step
    for _ in range(2000):
        f_lo = equation(s_lo)
        if f_lo >= 0:
            return brentq(equation, s_lo, s_hi, xtol=1e-12, rtol=1e-13)
        s_hi, f_hi = s_lo, f_lo
        s_lo -= step
    raise RuntimeError("Failed to find the nearest previous path point.")


def build_path_chain(path: TurnPath, s_head: float):
    s_values = np.empty(N_HANDLES)
    points = np.empty((N_HANDLES, 2))
    tangents = np.empty((N_HANDLES, 2))
    s_values[0] = s_head
    points[0], tangents[0] = path.point_tangent(s_head)
    for i in range(1, N_HANDLES):
        s_values[i] = next_path_coordinate(path, s_values[i - 1], handle_distance(i))
        points[i], tangents[i] = path.point_tangent(s_values[i])
    return s_values, points, tangents


def path_state(path: TurnPath, s_head: float, head_speed: float = 1.0):
    s_values, points, tangents = build_path_chain(path, s_head)
    speeds = velocities_from_tangents(points, tangents, head_speed)
    return s_values, points, tangents, speeds


def maximum_speed_amplification(path: TurnPath) -> dict:
    grid = np.linspace(-100, 100, 801)
    values = []
    handle_indices = []
    for s in grid:
        _, _, _, speeds = path_state(path, float(s), 1.0)
        index = int(np.argmax(np.abs(speeds)))
        values.append(float(abs(speeds[index])))
        handle_indices.append(index)
    values_array = np.array(values)
    candidate_indices = np.argsort(values_array)[-8:]
    best_value = -math.inf
    best_s = None
    best_handle = None
    for idx in candidate_indices:
        left = float(grid[max(0, idx - 1)])
        right = float(grid[min(len(grid) - 1, idx + 1)])

        def negative_amplification(s: float) -> float:
            _, _, _, speeds = path_state(path, s, 1.0)
            return -float(np.max(np.abs(speeds)))

        result = minimize_scalar(
            negative_amplification,
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-8},
        )
        value = -float(result.fun)
        if value > best_value:
            _, _, _, speeds = path_state(path, float(result.x), 1.0)
            best_value = value
            best_s = float(result.x)
            best_handle = int(np.argmax(np.abs(speeds)))
    return {
        "amplification": best_value,
        "s_head": best_s,
        "handle_index": best_handle,
        "max_head_speed": 2.0 / best_value,
        "grid_max": float(values_array.max()),
    }


def constraint_residual(points: np.ndarray) -> float:
    residuals = []
    for i in range(1, N_HANDLES):
        residuals.append(abs(np.linalg.norm(points[i] - points[i - 1]) - handle_distance(i)))
    return max(residuals)


def selected_handle_indices() -> list[int]:
    # head, body 1/51/101/151/201 front handles, tail rear handle
    return [0, 1, 51, 101, 151, 201, 223]


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_all(output: Path) -> dict:
    q1 = {}
    for t in (0, 60, 120, 180, 240, 300):
        theta = head_theta_at_time(t)
        _, points, _, speeds = spiral_state(theta, 0.55, 1.0)
        q1[str(t)] = {
            "positions": points[selected_handle_indices()],
            "speeds": np.abs(speeds[selected_handle_indices()]),
            "distance_residual": constraint_residual(points),
        }

    q2 = find_terminal_time()
    q2_summary = {
        "time": q2["time"],
        "clearance": q2["clearance"],
        "contact_pair": q2["pair"],
        "positions": q2["points"][selected_handle_indices()],
        "speeds": np.abs(q2["speeds"][selected_handle_indices()]),
        "distance_residual": constraint_residual(q2["points"]),
    }

    q3 = find_minimum_pitch()
    q3_summary = {key: value for key, value in q3.items() if key != "points"}
    q3_summary["distance_residual"] = constraint_residual(q3["points"])

    path = TurnPath()
    q4_geometry = path.geometry_report()
    q4 = {}
    for t in (-100, -50, 0, 50, 100):
        _, points, _, speeds = path_state(path, float(t), 1.0)
        q4[str(t)] = {
            "positions": points[selected_handle_indices()],
            "speeds": np.abs(speeds[selected_handle_indices()]),
            "distance_residual": constraint_residual(points),
        }

    q5 = maximum_speed_amplification(path)
    result = {
        "model": {
            "handles": N_HANDLES,
            "boards": N_BOARDS,
            "head_handle_distance": HEAD_HANDLE_DISTANCE,
            "body_handle_distance": BODY_HANDLE_DISTANCE,
            "board_width": BOARD_WIDTH,
        },
        "problem1": q1,
        "problem2": q2_summary,
        "problem3": q3_summary,
        "problem4_geometry": q4_geometry,
        "problem4": q4,
        "problem5": q5,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=serialize),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Independent solver for CUMCM 2024 Problem A")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("independent_results.json"),
    )
    args = parser.parse_args()
    result = run_all(args.output)
    print(json.dumps({
        "problem2": result["problem2"],
        "problem3": result["problem3"],
        "problem4_geometry": result["problem4_geometry"],
        "problem5": result["problem5"],
    }, ensure_ascii=False, indent=2, default=serialize))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
