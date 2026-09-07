"""Optimal-control problems: infinite-horizon, discounted, continuous-time.

    V(x) = min_u  integral_0^inf  e^{-rho t} l(x(t), u(t)) dt,
    xdot = f(x, u),   l(x, u) = q(x) + r(u).

The solvers only need callables f(X, u) -> (N, d) and running_cost(X, u) -> (N,)
that are vectorised over the state batch X (control scalar, or per-point array).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ControlProblem:
    f: Callable[[np.ndarray, float], np.ndarray]
    running_cost: Callable[[np.ndarray, float], np.ndarray]
    umin: float
    umax: float
    rho: float

    def control_grid(self, n_u: int) -> np.ndarray:
        return np.linspace(self.umin, self.umax, n_u)


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def double_integrator(q1=2.0, q2=0.4, r=0.2, rho=0.1, umax=4.0):
    """xdot1 = x2, xdot2 = u; l = 1/2 (q1 x1^2 + q2 x2^2 + r u^2).

    The discounted LQR solution is known in closed form, which makes this the
    validation case. Returns (problem, Q, R) with Q, R the weight matrices.
    """
    def f(x, u):
        out = np.empty_like(x, dtype=float)
        out[:, 0] = x[:, 1]
        out[:, 1] = u
        return out

    def running_cost(x, u):
        return 0.5 * (q1 * x[:, 0] ** 2 + q2 * x[:, 1] ** 2 + r * u * u)

    Q = np.diag([q1, q2])
    R = np.array([[r]])
    return ControlProblem(f, running_cost, -umax, umax, rho), Q, R


def pendulum(g=9.81, l=1.0, m=1.0, b=0.1, umax=3.5,
             w_theta=2.0, w_omega=0.1, r_u=0.1, rho=0.05):
    """Torque-limited pendulum swing-up.

    State x = (theta, omega) with theta measured from the *upright* equilibrium
    and periodic. umax < m g l, so the pendulum cannot be rotated up directly
    and must pump energy. l = 1/2 (w_theta theta^2 + w_omega omega^2 + r_u u^2).
    """
    k = 1.0 / (m * l * l)

    def f(x, u):
        out = np.empty_like(x, dtype=float)
        out[:, 0] = x[:, 1]
        out[:, 1] = (g / l) * np.sin(x[:, 0]) + k * u - b * x[:, 1]
        return out

    def running_cost(x, u):
        th = wrap_angle(x[:, 0])
        return 0.5 * (w_theta * th ** 2 + w_omega * x[:, 1] ** 2) + 0.5 * r_u * u * u

    return ControlProblem(f, running_cost, -umax, umax, rho)
