from __future__ import annotations

import math

import numpy as np


def clamp_target_xy(x: float, y: float, half_w: float, half_h: float, margin: float) -> tuple[float, float, bool]:
    lim_x = max(0.0, float(half_w) - max(0.0, float(margin)))
    lim_y = max(0.0, float(half_h) - max(0.0, float(margin)))
    xc = float(np.clip(float(x), -lim_x, lim_x))
    yc = float(np.clip(float(y), -lim_y, lim_y))
    return xc, yc, bool(abs(xc - float(x)) > 1e-9 or abs(yc - float(y)) > 1e-9)


def bounds_minmax_xy(origin_x: float, origin_y: float, bounds_w: float, bounds_h: float, margin: float) -> tuple[float, float, float, float]:
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    m = max(0.0, float(margin))
    x_min = float(origin_x) - half_w + m
    x_max = float(origin_x) + half_w - m
    y_min = float(origin_y) - half_h + m
    y_max = float(origin_y) + half_h - m
    if x_min > x_max:
        mid = 0.5 * (x_min + x_max)
        x_min = mid
        x_max = mid
    if y_min > y_max:
        mid = 0.5 * (y_min + y_max)
        y_min = mid
        y_max = mid
    return float(x_min), float(x_max), float(y_min), float(y_max)


def scan_center_offset_for_reference(reference: str, bounds_w: float, bounds_h: float, margin: float) -> tuple[float, float]:
    ref = str(reference or "center").strip().lower()
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    dx = max(0.0, half_w - max(0.0, float(margin)))
    dy = max(0.0, half_h - max(0.0, float(margin)))
    if ref == "sw":
        return float(dx), float(dy)
    if ref == "se":
        return float(-dx), float(dy)
    if ref == "nw":
        return float(dx), float(-dy)
    if ref == "ne":
        return float(-dx), float(-dy)
    return 0.0, 0.0


def clamp_target_xy_minmax(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[float, float, bool]:
    xc = float(np.clip(float(x), float(x_min), float(x_max)))
    yc = float(np.clip(float(y), float(y_min), float(y_max)))
    return xc, yc, bool(abs(xc - float(x)) > 1e-9 or abs(yc - float(y)) > 1e-9)


def append_dense_line(points: list[np.ndarray], a: np.ndarray, b: np.ndarray, ds: float) -> None:
    dxy = b[:2] - a[:2]
    seg_len = float(np.linalg.norm(dxy))
    if seg_len <= 1e-6:
        return
    n = max(1, int(np.ceil(seg_len / max(1e-3, float(ds)))))
    for k in range(1, n + 1):
        t = float(k / n)
        p = (1.0 - t) * a + t * b
        points.append(np.asarray(p, dtype=np.float32))


def append_dense_arc(
    points: list[np.ndarray],
    p_start: np.ndarray,
    p_end: np.ndarray,
    center_xy: np.ndarray,
    z: float,
    ccw: bool,
    ds: float,
) -> None:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    a0 = float(math.atan2(float(p_start[1]) - cy, float(p_start[0]) - cx))
    a1 = float(math.atan2(float(p_end[1]) - cy, float(p_end[0]) - cx))
    if ccw:
        while a1 <= a0:
            a1 += 2.0 * math.pi
        dtheta = a1 - a0
    else:
        while a1 >= a0:
            a1 -= 2.0 * math.pi
        dtheta = a1 - a0
    r = float(np.linalg.norm(np.asarray([float(p_start[0]) - cx, float(p_start[1]) - cy], dtype=np.float64)))
    arc_len = abs(dtheta) * max(1e-6, r)
    n = max(1, int(np.ceil(arc_len / max(1e-3, float(ds)))))
    for k in range(1, n + 1):
        t = float(k / n)
        a = a0 + dtheta * t
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        points.append(np.asarray([x, y, z], dtype=np.float32))


def build_lawnmower_targets(
    bounds_w: float,
    bounds_h: float,
    margin: float,
    step_len: float,
    alt_m: float,
    turn_style: str = "sharp",
    turn_radius_m: float = 2.0,
) -> list[np.ndarray]:
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    x_min = -half_w + float(margin)
    x_max = half_w - float(margin)
    y_min = -half_h + float(margin)
    y_max = half_h - float(margin)
    z = -abs(float(alt_m))
    if x_min >= x_max or y_min >= y_max:
        return [np.asarray([0.0, 0.0, z], dtype=np.float32)]

    lane_spacing = max(0.5, float(step_len))
    x_vals = list(np.arange(x_min, x_max + 1e-6, lane_spacing))
    if not x_vals or abs(float(x_vals[-1]) - x_max) > 1e-6:
        x_vals.append(float(x_max))
    poly = []
    for i, x in enumerate(x_vals):
        if i % 2 == 0:
            poly.append(np.asarray([x, y_min, z], dtype=np.float32))
            poly.append(np.asarray([x, y_max, z], dtype=np.float32))
        else:
            poly.append(np.asarray([x, y_max, z], dtype=np.float32))
            poly.append(np.asarray([x, y_min, z], dtype=np.float32))
    if len(poly) <= 1:
        return poly

    dense: list[np.ndarray] = [poly[0]]
    ds = max(0.2, float(step_len))
    turn_style_eff = str(turn_style).strip().lower()
    if turn_style_eff != "arc":
        for i in range(len(poly) - 1):
            append_dense_line(dense, poly[i], poly[i + 1], ds)
        return dense

    cursor = np.asarray(poly[0], dtype=np.float32)
    for i in range(1, len(poly) - 1):
        prev_p = np.asarray(poly[i - 1], dtype=np.float32)
        corner_p = np.asarray(poly[i], dtype=np.float32)
        next_p = np.asarray(poly[i + 1], dtype=np.float32)
        vin = np.asarray([float(corner_p[0] - prev_p[0]), float(corner_p[1] - prev_p[1])], dtype=np.float64)
        vout = np.asarray([float(next_p[0] - corner_p[0]), float(next_p[1] - corner_p[1])], dtype=np.float64)
        len_in = float(np.linalg.norm(vin))
        len_out = float(np.linalg.norm(vout))
        if len_in <= 1e-6 or len_out <= 1e-6:
            append_dense_line(dense, cursor, corner_p, ds)
            cursor = corner_p
            continue
        din = vin / len_in
        dout = vout / len_out
        dot = float(np.clip(np.dot(din, dout), -1.0, 1.0))
        is_corner = bool(abs(dot) < 0.25)
        r = min(max(0.0, float(turn_radius_m)), 0.45 * len_in, 0.45 * len_out)
        if (not is_corner) or r <= 1e-4:
            append_dense_line(dense, cursor, corner_p, ds)
            cursor = corner_p
            continue
        p1 = np.asarray([float(corner_p[0] - din[0] * r), float(corner_p[1] - din[1] * r), float(corner_p[2])], dtype=np.float32)
        p2 = np.asarray([float(corner_p[0] + dout[0] * r), float(corner_p[1] + dout[1] * r), float(corner_p[2])], dtype=np.float32)
        center_xy = np.asarray([float(corner_p[0] - din[0] * r + dout[0] * r), float(corner_p[1] - din[1] * r + dout[1] * r)], dtype=np.float64)
        append_dense_line(dense, cursor, p1, ds)
        cross_z = float(din[0] * dout[1] - din[1] * dout[0])
        append_dense_arc(dense, p_start=p1, p_end=p2, center_xy=center_xy, z=float(corner_p[2]), ccw=bool(cross_z > 0.0), ds=ds)
        cursor = p2

    append_dense_line(dense, cursor, np.asarray(poly[-1], dtype=np.float32), ds)
    return dense


def auto_step_len(scale: float) -> float:
    return 5.0 if float(scale) >= 2.0 else 3.0


def auto_accept_radius(scale: float) -> float:
    return 1.25 if float(scale) >= 2.0 else 0.75


def auto_vxy_cap(scale: float) -> float:
    return 1.5 if float(scale) >= 2.0 else 1.2
