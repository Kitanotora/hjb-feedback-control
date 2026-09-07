"""Efficient HJB solving for feedback controllers.

Semi-Lagrangian discretisation + Howard policy iteration.
"""
from .grid import Grid
from .problems import ControlProblem, double_integrator, pendulum, wrap_angle
from .solvers import (
    SolveResult,
    TableController,
    bellman_min,
    golden_refine,
    policy_evaluation,
    simulate,
    solve_policy_iteration,
    solve_value_iteration,
    trajectory_cost,
)

__all__ = [
    "Grid",
    "ControlProblem",
    "double_integrator",
    "pendulum",
    "wrap_angle",
    "SolveResult",
    "TableController",
    "bellman_min",
    "golden_refine",
    "policy_evaluation",
    "simulate",
    "solve_policy_iteration",
    "solve_value_iteration",
    "trajectory_cost",
]
