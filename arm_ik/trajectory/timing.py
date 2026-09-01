"""Time-parameterisation of a joint trajectory.

Given a joint path and per-joint velocity limits, produce a time-stamped
sequence ``(t, q, qd)`` suitable for a fixed-rate servo loop. A trapezoidal
velocity profile between waypoints respects the stated speed limits; if the
segment is too short to reach the limit speed it degrades to a triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRAJ_POINTS = 100


@dataclass(frozen=True)
class TimedTrajectory:
    """Timed joint-space samples at a fixed ``dt``."""

    times: np.ndarray
    q: np.ndarray
    qd: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample_at(self, t: float) -> np.ndarray:
        """Joint vector at time ``t`` via linear interpolation."""
        idx = np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 1)
        return self.q[idx]


_ACCEL_PASSES = 20


def time_parameterize(
    q_path: np.ndarray,
    velocity_limits: np.ndarray,
    dt: float = 0.02,
    accel: float | None = None,
) -> TimedTrajectory:
    """Time-parameterise a joint path using per-axis velocity limits.

    Args:
        q_path: ``(N, dof)`` joint waypoints, already interpolation-safe.
        velocity_limits: Per-joint maximum velocity in rad/s.
        dt: Sample period for the returned trajectory.
        accel: Optional joint acceleration limit in rad/s^2. Adjacent segments
            are stretched until the velocity change at every waypoint, divided
            by the mean duration of the two segments meeting there, is within
            ``accel``.

            This is a waypoint-level bound, not a bound on the sampled
            ``qd``: each segment is traversed at a constant velocity, so
            ``diff(qd) / dt`` on the returned grid still shows a step at every
            waypoint and will read far above ``accel``. What the limit buys is
            a path whose commanded velocity changes gently between waypoints;
            it is not a time-optimal trapezoid, and sharp corners get slowed
            down rather than rounded. Feed the result to a servo loop that
            interpolates between samples, or densify ``q_path`` first.
    """
    q_path = np.asarray(q_path, dtype=float)
    n, dof = q_path.shape
    velocity_limits = np.asarray(velocity_limits, dtype=float)
    if velocity_limits.ndim != 1:
        raise ValueError("velocity_limits必须是一维")
    if len(velocity_limits) != dof:
        raise ValueError("velocity_limits长度必须等于dof")
    if n < 2:
        raise ValueError("q_path至少需要两个路点")
    if accel is not None and accel <= 0:
        raise ValueError("accel必须为正")

    deltas = np.diff(q_path, axis=0)
    segment_dt = np.empty(n - 1)
    for i in range(n - 1):
        # Time to traverse this joint distance at the joint's velocity limit,
        # then take the slowest joint as the segment duration.
        t_seg = float(
            np.max(np.abs(deltas[i]) / np.maximum(velocity_limits, 1e-12))
        )
        segment_dt[i] = max(t_seg, dt)

    if accel is not None and n > 2:
        segment_dt = _limit_acceleration(deltas, segment_dt, accel)

    segment_start = np.concatenate([[0.0], np.cumsum(segment_dt)])
    total = float(segment_start[-1])
    samples = int(np.ceil(total / dt))
    times = np.arange(samples + 1) * dt
    q = np.empty((samples + 1, dof), dtype=float)
    qd = np.empty((samples + 1, dof), dtype=float)

    for i in range(n - 1):
        lo, hi = segment_start[i], segment_start[i + 1]
        mask = (times >= lo) & (times <= hi)
        t_local = (times[mask] - lo) / max(hi - lo, 1e-12)
        q[mask] = q_path[i] * (1 - t_local[:, None]) + q_path[i + 1] * t_local[:, None]
        vel = (q_path[i + 1] - q_path[i]) / max(hi - lo, 1e-12)
        qd[mask] = vel[None, :]

    # Fix the last sample exactly on the final waypoint if the grid overshoots.
    q[samples] = q_path[-1]
    qd[samples] = 0.0

    return TimedTrajectory(times=times, q=q, qd=qd)


def _limit_acceleration(
    deltas: np.ndarray,
    segment_dt: np.ndarray,
    accel: float,
) -> np.ndarray:
    """Stretch segments until the velocity change at each waypoint fits ``accel``.

    Each segment is traversed at a constant velocity ``deltas[i] / dt[i]``, so
    the whole velocity change lands at the waypoint between two segments. This
    relaxation grows the two adjacent segments until that change is within
    ``accel`` over their mean duration. It converges downward in violation and
    is capped by ``_ACCEL_PASSES`` so a pathological path cannot spin forever.
    """
    segment_dt = segment_dt.copy()
    for _ in range(_ACCEL_PASSES):
        worst = 0.0
        for i in range(len(segment_dt) - 1):
            v_in = deltas[i] / segment_dt[i]
            v_out = deltas[i + 1] / segment_dt[i + 1]
            window = 0.5 * (segment_dt[i] + segment_dt[i + 1])
            needed = float(np.max(np.abs(v_out - v_in))) / accel
            if needed > window:
                scale = needed / window
                segment_dt[i] *= scale
                segment_dt[i + 1] *= scale
                worst = max(worst, scale)
        if worst <= 1.0 + 1e-9:
            break
    return segment_dt
