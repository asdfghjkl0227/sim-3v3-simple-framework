"""Global path planner for walk_to.

The planner is intentionally small: it builds an 8-neighbor grid inside the
field, marks existing circular obstacles as blocked, and runs A*.  The caller
can then follow the first waypoint while keeping the existing velocity control.
"""

from __future__ import annotations

import heapq
import math

from ..framework.types import Context
from ..param import (
    GLOBAL_FIELD_MARGIN_M,
    GLOBAL_GRID_RESOLUTION_M,
    GLOBAL_OBSTACLE_MARGIN_M,
    HALF_SAFE_MARGIN_M,
)
from .obstacles import Obstacle
from .rules import segment_clear, bounds


def plan_global_path(
    context: Context,
    start: tuple[float, float],
    target: tuple[float, float],
    obstacles: list[Obstacle],
    *, own_half: bool = False, ball_access: bool = False,
) -> list[tuple[float, float]] | None:
    """Return an A* path from start to target, or None if no path is found."""
    min_x, max_x, min_y, max_y = _bounds(context)
    if ball_access:
        min_x,max_x,min_y,max_y = bounds(context,own_half,ball_access=True)
    if own_half:
        # 从对方半场返程时保留起点到中线的走廊；合法后不再规划越半场。
        max_x = min(max_x,max(-HALF_SAFE_MARGIN_M,start[0]))
    sx, sy = _clamp_point(start[0], start[1], min_x, max_x, min_y, max_y)
    tx, ty = _clamp_point(target[0], target[1], min_x, max_x, min_y, max_y)

    start_idx = _to_idx(sx, sy, min_x, min_y)
    goal_idx = _to_idx(tx, ty, min_x, min_y)
    if _blocked(tx, ty, obstacles):
        return None
    if start_idx == goal_idx:
        return [(tx, ty)] if segment_clear(start,target,obstacles,GLOBAL_OBSTACLE_MARGIN_M) else None

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start_idx))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_idx: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_idx:
            path = _reconstruct(came_from, current, min_x, min_y, (tx, ty))
            path[0] = start
            if all(segment_clear(a,b,obstacles,GLOBAL_OBSTACLE_MARGIN_M)
                   for a,b in zip(path,path[1:])):
                return path
            return None
        closed.add(current)

        for neighbor, step_cost in _neighbors(current):
            x, y = _to_xy(neighbor, min_x, min_y)
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue
            point = _to_xy(current,min_x,min_y)
            if own_half and point[0] > -HALF_SAFE_MARGIN_M and x > point[0]:
                continue
            if _blocked(x,y,obstacles) or not segment_clear(
                    point,(x,y),obstacles,GLOBAL_OBSTACLE_MARGIN_M):
                continue
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            heapq.heappush(
                open_heap,
                (tentative + _heuristic(neighbor, goal_idx), counter, neighbor),
            )

    return None


def _bounds(context: Context) -> tuple[float, float, float, float]:
    half_l = context.field.length / 2.0 - GLOBAL_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - GLOBAL_FIELD_MARGIN_M
    return -half_l, half_l, -half_w, half_w


def _clamp_point(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float]:
    return (max(min_x, min(max_x, x)), max(min_y, min(max_y, y)))


def _to_idx(x: float, y: float, min_x: float, min_y: float) -> tuple[int, int]:
    return (
        int(round((x - min_x) / GLOBAL_GRID_RESOLUTION_M)),
        int(round((y - min_y) / GLOBAL_GRID_RESOLUTION_M)),
    )


def _to_xy(idx: tuple[int, int], min_x: float, min_y: float) -> tuple[float, float]:
    return (
        min_x + idx[0] * GLOBAL_GRID_RESOLUTION_M,
        min_y + idx[1] * GLOBAL_GRID_RESOLUTION_M,
    )


def _neighbors(idx: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    x, y = idx
    out: list[tuple[tuple[int, int], float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            out.append(((x + dx, y + dy), cost))
    return out


def _blocked(x: float, y: float, obstacles: list[Obstacle]) -> bool:
    for obs in obstacles:
        if math.hypot(x - obs.x, y - obs.y) <= obs.radius + GLOBAL_OBSTACLE_MARGIN_M:
            return True
    return False


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
    min_x: float,
    min_y: float,
    target: tuple[float, float],
) -> list[tuple[float, float]]:
    indices = [current]
    while current in came_from:
        current = came_from[current]
        indices.append(current)
    indices.reverse()
    path = [_to_xy(idx, min_x, min_y) for idx in indices]
    path[-1] = target
    return _smooth_path(path)


def _smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    prev_dx = 0
    prev_dy = 0
    for i in range(1, len(path)):
        dx = _sign(path[i][0] - path[i - 1][0])
        dy = _sign(path[i][1] - path[i - 1][1])
        if i > 1 and (dx, dy) != (prev_dx, prev_dy):
            smoothed.append(path[i - 1])
        prev_dx, prev_dy = dx, dy
    smoothed.append(path[-1])
    return smoothed


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0
