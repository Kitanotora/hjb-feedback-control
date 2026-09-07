"""Example 3 -- efficiency: policy iteration vs. value iteration.

Both methods act on the *same* discrete Bellman operator (same grid, dt,
control candidates).  Value iteration is a contraction iteration with modulus
gamma = exp(-rho*dt); policy iteration is Newton's method on that operator.
We run value iteration until its Bellman residual is small enough that the
remaining fixed-point error is ~0.5% of the value scale, and compare work.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from hjb import Grid, pendulum, solve_policy_iteration, solve_value_iteration

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

problem = pendulum()
grid = Grid((-np.pi, -12.0), (np.pi, 12.0), (81, 81), periodic=(0,))
dt, n_u = 0.04, 33

print("policy iteration ...")
pi = solve_policy_iteration(problem, grid, dt, n_u=n_u, verbose=True)

gamma = np.exp(-problem.rho * dt)
v_scale = float(pi.V.max())
tol_vi = 0.005 * v_scale * (1.0 - gamma)  # fixed-point error bound ~ 0.5% of scale
print(f"\nvalue iteration to residual {tol_vi:.2e} "
      f"(fixed-point error bound = residual / (1-gamma), 1-gamma = {1 - gamma:.1e}) ...")
vi = solve_value_iteration(problem, grid, dt, n_u=n_u, tol=tol_vi,
                           max_iter=30000, verbose=True, check_every=500)

diff = float(np.max(np.abs(vi.V - pi.V))) / v_scale
rows = [
    ("method", "iterations", "seconds", "final residual", "value diff vs PI"),
    ("policy iteration", f"{pi.iterations}", f"{pi.seconds:.2f}",
     f"{pi.residuals[-1]:.1e}", "-"),
    ("value iteration", f"{vi.iterations}", f"{vi.seconds:.2f}",
     f"{vi.residuals[-1]:.1e}", f"{diff:.1e}"),
]
widths = [max(len(r[c]) for r in rows) for c in range(5)]
table = "\n".join("  ".join(r[c].ljust(widths[c]) for c in range(5)) for r in rows)
print("\n" + table)
print(f"speedup (wall time): {vi.seconds / pi.seconds:.0f}x   "
      f"iteration ratio: {vi.iterations / pi.iterations:.0f}x")
(FIG / "efficiency_summary.txt").write_text(table + "\n")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogy(np.arange(1, len(pi.residuals) + 1), pi.residuals, "o-", lw=1.8,
            label=f"policy iteration ({pi.iterations} it, {pi.seconds:.1f} s)")
ax.semilogy(np.arange(1, len(vi.residuals) + 1), vi.residuals, "-",
            lw=1.0, alpha=0.9, label=f"value iteration ({vi.iterations} it, {vi.seconds:.0f} s)")
ax.axhline(tol_vi, color="r", ls="--", lw=0.8, label="VI stopping residual")
ax.set_xlabel("iteration")
ax.set_ylabel("Bellman residual  ||T(V) - V||_inf")
ax.set_title("Pendulum, 81x81 grid: Howard PI vs. value iteration")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "efficiency.png", dpi=150)
print(f"\nfigure -> {FIG / 'efficiency.png'}")
