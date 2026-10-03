from __future__ import annotations

import numpy as np

from docking_sim.env.physics import WorldConfig, hull_aabb, port_aabb, wrap_angle


def _world_to_px(x: float, y: float, cfg: WorldConfig, width: int, height: int) -> tuple[int, int]:
    u = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
    v = (y - cfg.y_min) / (cfg.y_max - cfg.y_min)
    px = int(np.clip(u * (width - 1), 0, width - 1))
    py = int(np.clip((1.0 - v) * (height - 1), 0, height - 1))
    return px, py


def _fill_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(w, xb + 1)
    yb = min(h, yb + 1)
    if xb > xa and yb > ya:
        img[ya:yb, xa:xb] = color


def _fill_circle(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][mask] = color


def _draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, n + 1).astype(int)
    ys = np.linspace(y0, y1, n + 1).astype(int)
    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = color


def render_rgb(state, cfg: WorldConfig, width: int = 160, height: int = 128) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (8, 10, 22)

    hx0, hy0, hx1, hy1 = hull_aabb(cfg)
    p0 = _world_to_px(hx0, hy1, cfg, width, height)
    p1 = _world_to_px(hx1, hy0, cfg, width, height)
    _fill_rect(img, p0[0], p0[1], p1[0], p1[1], (72, 82, 104))

    px0, py0, px1, py1 = port_aabb(cfg)
    q0 = _world_to_px(px0, py1, cfg, width, height)
    q1 = _world_to_px(px1, py0, cfg, width, height)
    _fill_rect(img, q0[0], q0[1], q1[0], q1[1], (40, 170, 150))

    sx, sy = _world_to_px(state.x, state.y, cfg, width, height)
    scale = width / (cfg.x_max - cfg.x_min)
    radius = max(2, int(cfg.ship_radius * scale))
    _fill_circle(img, sx, sy, radius, (230, 232, 240))

    nose = 2.4 * radius
    theta = wrap_angle(state.theta)
    nx = int(sx + np.cos(theta) * nose)
    ny = int(sy - np.sin(theta) * nose)
    _draw_line(img, sx, sy, nx, ny, (255, 170, 70))
    return img
