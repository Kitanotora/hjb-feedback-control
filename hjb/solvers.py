"""Semi-Lagrangian discretisation of the stationary HJB + policy iteration.

Discretisation.  For the discounted infinite-horizon problem, one backward
Euler step of the Dynamic Programming principle along the dynamics gives the
semi-Lagrangian scheme, for any grid point x_i:

    V(x_i) = min_u [ dt * l(x_i, u) + gamma * V_interp(x_i + dt * f(x_i, u)) ],
    gamma = exp(-rho * dt).

Properties (Falcone & Ferretti, 2013):
  * monotone and unconditionally stable -- dt is a pure accuracy trade-off,
    there is no CFL restriction;
  * it is exactly the Bellman operator of a Markov decision process whose
    transition matrix is the (row-stochastic) interpolation-weight matrix, so
    Howard's policy iteration applies and terminates in finitely many steps.

Policy iteration (Howard).  Fix a policy u, solve the *linear* system that
evaluates it, then improve pointwise:

    evaluate:  V = dt * l(x, u) + gamma * P(u) V      (sparse LU)
    improve :  u+ = argmin_u [ dt * l(x, u) + gamma * V_interp(x + dt f(x, u)) ]

This is Newton's method on the Bellman operator: a handful of sparse solves
instead of the ~1/(rho*dt) contraction sweeps that value iteration needs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .problems import ControlProblem


@dataclass
class SolveResult:
    V: np.ndarray          # value function on the grid (grid.shape)
    U: np.ndarray          # optimal control on the grid (grid.shape)
    dt: float
    iterations: int
    residuals: list = field(default_factory=list)
    seconds: float = 0.0
    method: str = ""


def bellman_min(problem: ControlProblem, grid: Grid, V, us, dt):
    """Apply the discrete Bellman operator: returns (T V, argmin U)."""
    X = grid.points()
    Vf = np.asarray(V, dtype=float).ravel()
    gamma = np.exp(-problem.rho * dt)
    costs = np.empty((len(us), grid.size))
    for k, u in enumerate(us):
        foot = X + dt * problem.f(X, u)
        costs[k] = dt * problem.running_cost(X, u) + gamma * grid.interp(Vf, foot)
    k_best = np.argmin(costs, axis=0)
    TV = costs[k_best, np.arange(grid.size)]
    return TV, us[k_best]


def policy_evaluation(problem: ControlProblem, grid: Grid, U, dt):
    """Solve the linear system that evaluates a fixed policy U (flat array)."""
    X = grid.points()
    gamma = np.exp(-problem.rho * dt)
    foot = X + dt * problem.f(X, U)
    idx, w = grid.interp_indices(foot)
    N = grid.size
    rows = np.repeat(np.arange(N), idx.shape[1])
    P = sp.coo_matrix((w.ravel(), (rows, idx.ravel())), shape=(N, N)).tocsr()
    A = (sp.eye(N, format="csc") - gamma * P).tocsc()
    c = dt * problem.running_cost(X, U)
    return spla.splu(A).solve(c)


def golden_refine(problem: ControlProblem, grid: Grid, V, U, dt, span, iters=40):
    """Vectorised golden-section polish of the policy U within +-span."""
    X = grid.points()
    Vf = np.asarray(V, dtype=float).ravel()
    gamma = np.exp(-problem.rho * dt)

    def h(u):
        foot = X + dt * problem.f(X, u)
        return dt * problem.running_cost(X, u) + gamma * grid.interp(Vf, foot)

    a = np.maximum(problem.umin, U - span)
    b = np.minimum(problem.umax, U + span)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iters):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        left = h(c) <= h(d)
        b = np.where(left, d, b)
        a = np.where(left, a, c)
    return 0.5 * (a + b)


def solve_policy_iteration(problem, grid, dt, n_u=65, max_iter=100, tol=1e-7,
                           U0=None, verbose=False):
    """Howard policy iteration on the semi-Lagrangian Bellman equation."""
    us = problem.control_grid(n_u)
    U = np.zeros(grid.size) if U0 is None else np.asarray(U0, float).ravel().copy()
    residuals = []
    t0 = time.perf_counter()
    it = 0
    for it in range(1, max_iter + 1):
        V = policy_evaluation(problem, grid, U, dt)
        TV, U_new = bellman_min(problem, grid, V, us, dt)
        res = float(np.max(np.abs(TV - V)))
        pol_change = float(np.max(np.abs(U_new - U)))
        residuals.append(res)
        if verbose:
            print(f"  PI iter {it:3d}: residual {res:.3e}, policy change {pol_change:.3e}")
        U = U_new
        if pol_change == 0.0 or res < tol:
            break
    V = policy_evaluation(problem, grid, U, dt)
    return SolveResult(V=V.reshape(grid.shape), U=U.reshape(grid.shape), dt=dt,
                       iterations=it, residuals=residuals,
                       seconds=time.perf_counter() - t0, method="policy iteration")


def solve_value_iteration(problem, grid, dt, n_u=65, max_iter=100000, tol=1e-6,
                          verbose=False, check_every=1000):
    """Plain value iteration on the same Bellman operator (for comparison)."""
    us = problem.control_grid(n_u)
    V = np.zeros(grid.size)
    residuals = []
    t0 = time.perf_counter()
    it = 0
    for it in range(1, max_iter + 1):
        TV, U = bellman_min(problem, grid, V, us, dt)
        res = float(np.max(np.abs(TV - V)))
        V = TV
        residuals.append(res)
        if verbose and it % check_every == 0:
            print(f"  VI iter {it:6d}: residual {res:.3e}")
        if res < tol:
            break
    return SolveResult(V=V.reshape(grid.shape), U=U.reshape(grid.shape), dt=dt,
                       iterations=it, residuals=residuals,
                       seconds=time.perf_counter() - t0, method="value iteration")


class TableController:
    """Feedback law u(x): multilinear interpolation of the policy table."""

    def __init__(self, grid: Grid, U):
        self.grid = grid
        self.U = np.asarray(U, dtype=float).ravel()

    def __call__(self, x):
        scalar = np.ndim(x) == 1
        vals = self.grid.interp(self.U, np.atleast_2d(np.asarray(x, dtype=float)))
        return float(vals[0]) if scalar else vals


def simulate(problem, controller, x0, T, h=0.005):
    """Closed-loop RK4 rollout with zero-order-hold control. ts, X (M, d), U (M,)."""
    x = np.asarray(x0, dtype=float).copy()
    n = int(round(T / h))
    ts = np.arange(n + 1) * h
    X = np.empty((n + 1, x.size))
    U = np.empty(n + 1)
    X[0] = x
    for i in range(n):
        u = float(controller(x.reshape(1, -1))[0])
        U[i] = u

        def g(xx):
            return problem.f(xx.reshape(1, -1), u)[0]

        k1 = g(x)
        k2 = g(x + 0.5 * h * k1)
        k3 = g(x + 0.5 * h * k2)
        k4 = g(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        X[i + 1] = x
    U[n] = float(controller(x.reshape(1, -1))[0])
    return ts, X, U


def trajectory_cost(problem, ts, X, U, rho=None):
    """Discounted cost integral along a simulated trajectory (trapezoid rule)."""
    rho = problem.rho if rho is None else rho
    ell = problem.running_cost(X[:-1], U[:-1])
    disc = np.exp(-rho * ts[:-1])
    return float(np.trapezoid(ell * disc, ts[:-1]))
