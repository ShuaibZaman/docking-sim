from __future__ import annotations

import numpy as np

from docking_sim.env.physics import (
    PortSpec,
    StationPose,
    WorldConfig,
    port_list,
    port_world_center,
    station_center,
    wrap_angle,
)

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 128


def _world_to_px(x: float, y: float, cfg: WorldConfig, width: int, height: int) -> tuple[int, int]:
    u = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
    v = (y - cfg.y_min) / (cfg.y_max - cfg.y_min)
    px = int(np.clip(u * (width - 1), 0, width - 1))
    py = int(np.clip((1.0 - v) * (height - 1), 0, height - 1))
    return px, py


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


def _fill_rotated_rect(
    img: np.ndarray,
    cx: float,
    cy: float,
    width_world: float,
    height_world: float,
    theta: float,
    color: tuple[int, int, int],
    cfg: WorldConfig,
    width: int,
    height: int,
) -> None:
    ys, xs = np.ogrid[0:height, 0:width]
    world_x = cfg.x_min + (xs + 0.5) / width * (cfg.x_max - cfg.x_min)
    world_y = cfg.y_max - (ys + 0.5) / height * (cfg.y_max - cfg.y_min)
    dx = world_x - cx
    dy = world_y - cy
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    mask = (np.abs(local_x) <= width_world / 2.0) & (np.abs(local_y) <= height_world / 2.0)
    img[mask] = color


def render_rgb(
    state,
    cfg: WorldConfig,
    pose: StationPose | None = None,
    asteroids: list[tuple[float, float, float]] | None = None,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    ports: tuple[PortSpec, ...] | None = None,
    active_port: int = 0,
) -> np.ndarray:
    pose = pose or StationPose()
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (8, 10, 22)

    for rock_x, rock_y, radius in asteroids or []:
        sx, sy = _world_to_px(rock_x, rock_y, cfg, width, height)
        scale = width / (cfg.x_max - cfg.x_min)
        _fill_circle(img, sx, sy, max(2, int(radius * scale)), (150, 136, 112))

    hull_x, hull_y = station_center(cfg, pose)
    _fill_rotated_rect(img, hull_x, hull_y, cfg.hull_w, cfg.hull_h, pose.theta, (72, 82, 104), cfg, width, height)

    for index, spec in enumerate(ports or port_list(cfg)):
        port_x, port_y = port_world_center(cfg, pose, spec)
        color = (40, 170, 150) if index == active_port else (24, 90, 84)
        _fill_rotated_rect(img, port_x, port_y, spec.w, spec.h, pose.theta, color, cfg, width, height)

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
