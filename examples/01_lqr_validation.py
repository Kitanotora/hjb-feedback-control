"""Example 1 -- validation against the exact discounted LQR solution.

For xdot = A x + B u with l = 1/2 (x'Qx + u'Ru) e^{-rho t}, the HJB solution
is V(x) = 1/2 x'Px where P solves the discounted algebraic Riccati equation

    (A - rho/2 I)'P + P(A - rho/2 I) - P B R^-1 B' P + Q = 0,

with feedback u*(x) = -R^-1 B'Px.  Weights are chosen so |u_exact| <= 3.9 on
the grid while u_max = 6, i.e. the box constraint is inactive and the Riccati
solution is the exact solution of the constrained HJB.

Error structure: the semi-Lagrangian scheme is locally consistent to
O(dt^2 + dx^2), but for the stationary discounted problem the local residual
accumulates over ~1/(rho*dt) steps, so the global error scales like
O(dt + dx^2/dt) and dt, dx must be refined *together*.  The convergence study
below does exactly that.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.linalg import solve_continuous_are

from hjb import (
    Grid,
    double_integrator,
    golden_refine,
    policy_evaluation,
    solve_policy_iteration,
)

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

rho = 0.1
problem, Q, R = double_integrator(q1=1.0, q2=1.0, r=1.0, rho=rho, umax=6.0)
r = float(R[0, 0])

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
P = solve_continuous_are(A - 0.5 * rho * np.eye(2), B, Q, R)


def V_exact(x):
    return 0.5 * np.einsum("ij,jk,ik->i", x, P, x)


def u_exact(x):
    return -(P[0, 1] * x[:, 0] + P[1, 1] * x[:, 1]) / r


def solve(grid, dt, n_u=65):
    res = solve_policy_iteration(problem, grid, dt, n_u=n_u)
    U = golden_refine(problem, grid, res.V, res.U.ravel(), dt,
                      span=(problem.umax - problem.umin) / n_u)
    V = policy_evaluation(problem, grid, U, dt)
    return V.ravel(), U, res


# ---- accuracy at the working resolution -------------------------------------
grid = Grid((-2.0, -2.0), (2.0, 2.0), (121, 121))
X = grid.points()
central = np.all(np.abs(X) <= 1.5, axis=1)
v_scale = V_exact(X[central]).max()

dt = 0.02
V_num, U_num, res = solve(grid, dt)
err_V = np.abs(V_num[central] - V_exact(X[central])) / v_scale
err_U = np.abs(U_num[central] - u_exact(X[central])) / problem.umax
print(f"PI: {res.iterations} iterations, {res.seconds:.2f} s, "
      f"final residual {res.residuals[-1]:.2e}")
print(f"value error   (central, rel. L-inf): {err_V.max():.3e}")
print(f"policy error  (central, rel. L-inf): {err_U.max():.3e}")

# ---- joint (dt, dx) convergence ----------------------------------------------
pairs = [(0.08, 61), (0.04, 121), (0.02, 241), (0.01, 481)]
dts, errs = [], []
for h, n in pairs:
    g = Grid((-2.0, -2.0), (2.0, 2.0), (n, n))
    Xg = g.points()
    cen = np.all(np.abs(Xg) <= 1.5, axis=1)
    V_h, _, r_h = solve(g, h)
    e = np.abs(V_h[cen] - V_exact(Xg[cen])).max() / v_scale
    dts.append(h)
    errs.append(e)
    print(f"dt = {h:5.3f}, n = {n:4d}: rel. value error {e:.3e} "
          f"({r_h.iterations} PI iterations, {r_h.seconds:.1f} s)")

slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
print(f"empirical convergence order (joint dt,dx refinement): {slope:.2f} (theory: 1)")

# ---- figures ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

ax = axes[0]
rel = np.abs(V_num - V_exact(X)) / v_scale
im = ax.imshow(rel.reshape(grid.shape).T, origin="lower",
               extent=(-2, 2, -2, 2), cmap="viridis")
ax.set_title("HJB value error  |V_num - V_exact| / max|V_exact|")
plt.colorbar(im, ax=ax)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

ax = axes[1]
rel_u = np.abs(U_num - u_exact(X)) / problem.umax
im = ax.imshow(rel_u.reshape(grid.shape).T, origin="lower",
               extent=(-2, 2, -2, 2), cmap="viridis")
ax.set_title("feedback error  |u_num - u_exact| / u_max")
plt.colorbar(im, ax=ax)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

ax = axes[2]
ax.loglog(dts, errs, "o-", label="measured")
ax.loglog(dts, [errs[-1] * (h / dts[-1]) for h in dts], "k--", label="slope 1")
ax.set_xlabel("dt  (with dx = dt/3 refined jointly)")
ax.set_ylabel("relative value error")
ax.set_title(f"joint (dt, dx) convergence (order {slope:.2f})")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Validation: HJB (policy iteration) vs. exact discounted LQR solution")
fig.tight_layout()
fig.savefig(FIG / "lqr_validation.png", dpi=150)
print(f"\nfigure -> {FIG / 'lqr_validation.png'}")
