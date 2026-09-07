"""Example 2 -- feedback controller for torque-limited pendulum swing-up.

The pendulum must swing up from hanging (theta = pi) to upright (theta = 0)
with |u| <= 3.5 < m g l = 9.81: it cannot be powered straight up and must pump
energy.  A local LQR controller about upright fails from hanging when
saturated; the HJB feedback controller u*(x) derived from the global value
function swings up near-optimally.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.linalg import solve_continuous_are

from hjb import (
    Grid,
    TableController,
    golden_refine,
    pendulum,
    policy_evaluation,
    simulate,
    solve_policy_iteration,
    trajectory_cost,
    wrap_angle,
)

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

problem = pendulum()
grid = Grid((-np.pi, -12.0), (np.pi, 12.0), (161, 161), periodic=(0,))
dt = 0.02

print("solving HJB by policy iteration ...")
res = solve_policy_iteration(problem, grid, dt, n_u=65, verbose=True)
U = golden_refine(problem, grid, res.V, res.U.ravel(), dt,
                  span=(problem.umax - problem.umin) / 65)
V = policy_evaluation(problem, grid, U, dt)
print(f"PI: {res.iterations} iterations, {res.seconds:.1f} s; "
      f"V(hanging) = {V.reshape(grid.shape)[grid.n[0] // 2, 0]:.2f}")

hjb = TableController(grid, U)

# ---- baseline: saturated LQR about upright ------------------------------------
A = np.array([[0.0, 1.0], [9.81, -0.1]])
B = np.array([[0.0], [1.0]])
P = solve_continuous_are(A, B, np.diag([2.0, 0.1]), np.array([[0.1]]))
K = (P @ B / 0.1).ravel()


def lqr(x):
    xw = x.copy()
    xw[:, 0] = wrap_angle(xw[:, 0])
    return np.clip(-(xw @ K), problem.umin, problem.umax)


class Callable:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        v = self.fn(np.atleast_2d(x))
        return float(v[0]) if np.ndim(x) == 1 else v


lqr_c = Callable(lqr)

# ---- closed-loop comparison from hanging --------------------------------------
T = 12.0
starts = [(np.pi, 0.0), (2.6, 0.0), (np.pi, 4.0)]
ts, Xh, Uh = simulate(problem, hjb, starts[0], T)
ts2, Xl, Ul = simulate(problem, lqr_c, starts[0], T)


def settle_time(ts, X, tol_th=0.1, tol_w=0.3):
    ok = (np.abs(wrap_angle(X[:, 0])) < tol_th) & (np.abs(X[:, 1]) < tol_w)
    for k in range(len(ts)):
        if ok[k:].all():
            return ts[k]
    return np.inf


print(f"\nfrom hanging (pi, 0), horizon {T} s:")
print(f"  HJB  : settles at t = {settle_time(ts, Xh):6.2f} s, "
      f"max|u| = {np.abs(Uh).max():.2f}, "
      f"discounted cost = {trajectory_cost(problem, ts, Xh, Uh):.2f}")
print(f"  LQR  : settles at t = {settle_time(ts2, Xl):6.2f} s, "
      f"max|u| = {np.abs(Ul).max():.2f}, "
      f"discounted cost = {trajectory_cost(problem, ts2, Xl, Ul):.2f}")

# ---- figures -------------------------------------------------------------------
th_ax, w_ax = grid.axes()
TH, W = np.meshgrid(th_ax, w_ax, indexing="ij")


def wrap01(a):
    return (a - a.min()) / (a.max() - a.min())


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axes[0]
vmax = np.percentile(V, 99.5)
im = ax.pcolormesh(TH, W, np.minimum(V.reshape(grid.shape), vmax),
                   cmap="viridis", shading="auto")
ax.contour(TH, W, V.reshape(grid.shape), levels=14, colors="w", linewidths=0.4, alpha=0.5)
plt.colorbar(im, ax=ax, label="V (color clipped at 99.5th pct)")
ax.set_title("value function $V(\\theta,\\omega)$")
ax.set_xlabel(r"$\theta$ [rad]")
ax.set_ylabel(r"$\omega$ [rad/s]")

ax = axes[1]
im = ax.pcolormesh(TH, W, U.reshape(grid.shape), cmap="RdBu_r",
                   vmin=problem.umin, vmax=problem.umax, shading="auto")
plt.colorbar(im, ax=ax, label="u*(x)")
ax.set_title("feedback law $u^*(x)$")
ax.set_xlabel(r"$\theta$ [rad]")
ax.set_ylabel(r"$\omega$ [rad/s]")

fig.suptitle("Pendulum swing-up: HJB solution (semi-Lagrangian + policy iteration)")
fig.tight_layout()
fig.savefig(FIG / "pendulum_solution.png", dpi=150)

fig = plt.figure(figsize=(11.5, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

ax = fig.add_subplot(gs[0, 0])
ax.plot(ts, wrap_angle(Xh[:, 0]), label="HJB feedback", lw=1.8)
ax.plot(ts2, wrap_angle(Xl[:, 0]), label="saturated LQR", lw=1.4, alpha=0.8)
ax.axhline(0.0, color="k", lw=0.6, ls=":")
ax.set_ylabel(r"$\theta$ (wrapped) [rad]")
ax.set_title("swing-up from hanging")
ax.legend()
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(ts, Xh[:, 1], label="HJB feedback", lw=1.8)
ax.plot(ts2, Xl[:, 1], label="saturated LQR", lw=1.4, alpha=0.8)
ax.axhline(0.0, color="k", lw=0.6, ls=":")
ax.set_ylabel(r"$\omega$ [rad/s]")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
ax.plot(ts, Uh, label="HJB feedback", lw=1.8)
ax.plot(ts2, Ul, label="saturated LQR", lw=1.4, alpha=0.8)
ax.axhline(problem.umax, color="r", lw=0.8, ls="--")
ax.axhline(problem.umin, color="r", lw=0.8, ls="--")
ax.set_ylabel(r"u [N$\cdot$m]")
ax.set_xlabel("t [s]")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.contour(TH, W, V.reshape(grid.shape), levels=14, cmap="viridis", alpha=0.35)
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(starts)))
for c, x0 in zip(colors, starts):
    _, Xs, _ = simulate(problem, hjb, x0, 8.0)
    ax.plot(wrap_angle(Xs[:, 0]), Xs[:, 1], color=c, lw=1.6)
    ax.plot(wrap_angle(Xs[0, 0]), Xs[0, 1], "o", color=c, ms=5)
_, Xl2, _ = simulate(problem, lqr_c, (np.pi, 0.0), 8.0)
ax.plot(wrap_angle(Xl2[:, 0]), Xl2[:, 1], "--", color="gray", lw=1.2, label="sat. LQR")
ax.plot(0, 0, "k*", ms=12)
ax.set_xlabel(r"$\theta$ [rad]")
ax.set_ylabel(r"$\omega$ [rad/s]")
ax.set_xlim(-np.pi, np.pi)
ax.set_title("phase portrait on value contours")
ax.legend()

fig.tight_layout()
fig.savefig(FIG / "pendulum_closedloop.png", dpi=150)
print(f"\nfigures -> {FIG / 'pendulum_solution.png'}, {FIG / 'pendulum_closedloop.png'}")
