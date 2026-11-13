"""Analytic double-gyre flow on the domain [0, 2] x [0, 1].

Stream function (Shadden, Lekien & Marsden, 2005):

    psi(x, y, t) = A * sin(pi * f(x, t)) * sin(pi * y)
    f(x, t)      = a(t) * x^2 + b(t) * x
    a(t)         = eps * sin(omega * t)
    b(t)         = 1 - 2 * eps * sin(omega * t)

Velocities (incompressible by construction, u = dpsi/dy, v = -dpsi/dx):

    u =  pi * A * sin(pi * f) * cos(pi * y)
    v = -pi * A * cos(pi * f) * (2 * a(t) * x + b(t)) * sin(pi * y)

The two gyres are the cells 0 < f < 1 (left) and 1 < f < 2 (right); the
separatrix f = 1 oscillates around x = 1, so the gyre centers move and the
cells periodically exchange fluid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DoubleGyre:
    """Parameters of the double-gyre flow (defaults follow Shadden et al.)."""

    A: float = 0.1
    eps: float = 0.25
    omega: float = 2.0 * np.pi / 10.0  # forcing period T = 10

    def a(self, t: float) -> float:
        return self.eps * np.sin(self.omega * t)

    def b(self, t: float) -> float:
        return 1.0 - 2.0 * self.eps * np.sin(self.omega * t)

    def f(self, x: ArrayLike, t: float) -> FloatArray:
        x = np.asarray(x, dtype=float)
        return self.a(t) * x**2 + self.b(t) * x

    def streamfunction(self, x: ArrayLike, y: ArrayLike, t: float) -> FloatArray:
        """psi at (x, y, t); x and y broadcast against each other."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.A * np.sin(np.pi * self.f(x, t)) * np.sin(np.pi * y)

    def velocity(self, x: ArrayLike, y: ArrayLike, t: float) -> tuple[FloatArray, FloatArray]:
        """Velocity (u, v) at (x, y, t); x and y broadcast against each other."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fv = self.f(x, t)
        u = np.pi * self.A * np.sin(np.pi * fv) * np.cos(np.pi * y)
        v = -np.pi * self.A * np.cos(np.pi * fv) * (2.0 * self.a(t) * x + self.b(t)) * np.sin(np.pi * y)
        return u, v

    def advect(self, points: ArrayLike, t0: float, dt: float, steps: int) -> FloatArray:
        """RK4-advect `points` (shape (N, 2)) from t0 to t0 + steps * dt.

        Needed for particle plots and later FTLE experiments; the tracking
        pipeline itself works from the closed-form centers.
        """
        pts = np.array(points, dtype=float, copy=True)
        t = float(t0)
        for _ in range(steps):
            k1 = self._velocity_at(pts, t)
            k2 = self._velocity_at(pts + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._velocity_at(pts + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._velocity_at(pts + dt * k3, t + dt)
            pts = pts + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt
        return pts

    def _velocity_at(self, pts: FloatArray, t: float) -> FloatArray:
        u, v = self.velocity(pts[..., 0], pts[..., 1], t)
        return np.stack([u, v], axis=-1)
