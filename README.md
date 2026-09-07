# Solving the HJB equation efficiently for feedback controllers

For a continuous-time optimal control problem

```
V(x) = min_u  ∫₀^∞ e^{-ρt} [ q(x) + r(u) ] dt,    ẋ = f(x, u),   u ∈ [u_min, u_max],
```

the stationarity condition of dynamic programming is the Hamilton–Jacobi–Bellman
equation

```
ρV(x) = min_u { q(x) + r(u) + ∇V(x)·f(x, u) },
```

and the feedback controller is recovered from the value function as

```
u*(x) = argmin_u { r(u) + ∇V(x)·f(x, u) }.
```

The value function is the *global* object: it encodes stability, saturation and
multi-modal behaviour (e.g. swing-up) that local methods like LQR miss. The
difficulty is that the HJB equation is a nonlinear, degenerate elliptic PDE, and
naive solution methods are slow. This repo implements the efficient standard
answer and demonstrates it end-to-end.

## The method: semi-Lagrangian scheme + Howard policy iteration

**Discretisation (semi-Lagrangian).** One backward-Euler step of the DP
principle along the dynamics gives, at every grid node `x_i` (Falcone &
Ferretti, 2013):

```
V(x_i) = min_u [ dt·ℓ(x_i, u) + γ·V_interp(x_i + dt·f(x_i, u)) ],   γ = e^{-ρ·dt}.
```

* **Unconditionally stable, no CFL restriction** — `dt` is a pure accuracy
  trade-off, unlike explicit finite-difference HJB schemes.
* Monotone, and exactly the Bellman operator of a Markov decision process whose
  transition matrix is the (row-stochastic) multilinear interpolation-weight
  matrix.
* Footpoints leaving the box are handled by clamping on non-periodic axes and by
  exact wrap-around on periodic axes (angles).

**Solver (policy iteration / Howard's algorithm).** The Bellman fixed point is
*not* found by contraction sweeps (value iteration, which needs
`O(1/(ρ·dt))` sweeps). Instead, alternate:

```
evaluate:  V = dt·ℓ(x, u) + γ·P(u) V        ← one sparse LU solve (I − γP)V = c
improve :  u = argmin_u [ dt·ℓ(x, u) + γ·V_interp(x + dt·f(x, u)) ]   ← pointwise
```

This is Newton's method on the Bellman operator: for the finite action set it
terminates in finitely many steps, and in practice needs ~6–20 iterations
regardless of how small `ρ·dt` is. On top of this one can add Gauss–Seidel
sweeps, multigrid or domain decomposition; none are needed at the scales here.

**Controller.** The feedback law is the policy table `u*(x_i)` plus multilinear
interpolation (`TableController`), evaluated online in microseconds. For
control-affine dynamics and quadratic `r`, the same law can be written in closed
form as `u* = clip(−R⁻¹g(x)ᵀ∇V̂(x))` with the interpolated gradient.

## Results (this machine, Apple silicon laptop)

**Validation — discounted LQR, double integrator** (`examples/01_lqr_validation.py`)
against the exact discounted Riccati solution `V = ½xᵀPx`, `u* = −R⁻¹BᵀPx`
(box constraint provably inactive):

| grid | dt | PI iterations | time | value error | feedback error |
|---|---|---|---|---|---|
| 121×121 | 0.02 | 13 | 0.9 s | 2.5 % | 2.4 % |
| 481×481 | 0.01 | 17 | 27 s | 1.0 % | — |

Joint `(dt, dx)` refinement gives empirical convergence order **0.98**
(theory: 1). Note the error structure: local consistency is `O(dt² + dx²)`, but
the *stationary* problem accumulates the residual over `~1/(ρ·dt)` steps, so the
global error scales like `O(dt + dx²/dt)` — refine `dt` and `dx` together, and
expect the constant to grow as `1/ρ` (longer horizon = harder).

**Pendulum swing-up under torque saturation** (`examples/02_pendulum_swingup.py`):
`mgl = 9.81 > u_max = 3.5`, so the pendulum must pump energy. HJB solved on a
161×161 grid (θ periodic), `dt = 0.02`, in **8 policy iterations / 1.7 s**,
converging to residual `2·10⁻¹⁴` (exact finite termination of Howard's
algorithm). From hanging:

| controller | settles (|θ|<0.1, \|ω\|<0.3) | max \|u\| | discounted cost |
|---|---|---|---|
| HJB feedback | 3.65 s | 3.50 | **19.5** |
| saturated LQR about upright | never | 3.50 | 75.1 (3.8× worse) |

**Efficiency — PI vs VI on the identical discrete operator**
(`examples/03_efficiency.py`, pendulum, 81×81, `dt = 0.04`, 33 control
candidates):

| method | iterations | seconds | final residual |
|---|---|---|---|
| policy iteration | 6 | 0.20 | 2·10⁻¹⁴ |
| value iteration | 782 | 11.7 | 1.9·10⁻⁴ |

**≈60× faster in wall time** (57–59× across runs; 130× fewer iterations) *and*
PI reaches machine
precision while VI is stopped at a residual giving ~0.5 % value accuracy. The
gap widens as `ρ·dt → 0` (VI needs `~1/(ρ·dt)` sweeps; PI is unaffected) — i.e.
exactly in the long-horizon regime that matters for regulation.

Figures: `figures/lqr_validation.png`, `figures/pendulum_solution.png`,
`figures/pendulum_closedloop.png`, `figures/efficiency.png`.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
PYTHONPATH=. .venv/bin/python examples/01_lqr_validation.py
PYTHONPATH=. .venv/bin/python examples/02_pendulum_swingup.py
PYTHONPATH=. .venv/bin/python examples/03_efficiency.py
```

Solve your own problem (any vectorised `f(X, u)` and `ℓ(X, u)`):

```python
from hjb import Grid, ControlProblem, solve_policy_iteration, TableController

grid = Grid(lo, hi, n, periodic=(0,))          # rectangular grid, axis 0 wraps
res  = solve_policy_iteration(problem, grid, dt=0.02, n_u=65, verbose=True)
u    = TableController(grid, res.U)            # u(x) in ~microseconds
```

## When to use what

* **State dim ≤ 3–4** (grid up to ~10⁷ nodes): this approach — semi-Lagrangian +
  policy iteration. Each iteration is a sparse solve with ~`2^d` nonzeros/row.
* **Dynamics affine in u, quadratic control cost**: do the inner minimisation in
  closed form instead of scanning candidates (`u* = clip(−R⁻¹g(x)ᵀ∇V)`) — same
  algorithm, cheaper improvement step.
* **Higher dims**: grid methods hit the curse of dimensionality. Then use
  trajectory methods (iLQR/DDP, direct transcription) recomputed as MPC for
  feedback; or function approximation for `V` (supervised fitting of a grid /
  self-supervised neural HJB residual minimisation); or, for polynomial systems,
  sum-of-squares relaxations of the HJB inequality.
* **Minimum-time / reachability / constraints as targets**: same machinery with
  `ℓ = 1` (or level-set reachability toolboxes).
* **Stochastic dynamics**: add the diffusion term `½tr(σσᵀ∇²V)` with a wide
  stencil — policy iteration still applies.

## Layout

```
hjb/grid.py       rectangular grids, periodic axes, multilinear interpolation
hjb/problems.py   problem definitions (double integrator, pendulum) + spec
hjb/solvers.py    Bellman operator, policy evaluation (sparse LU),
                  policy iteration, value iteration, golden-section policy
                  polish, closed-loop simulator, TableController
examples/         validation, swing-up demo, efficiency benchmark
figures/          generated figures
```

## References

* R. Bellman, *Dynamic Programming*, 1957.
* R. A. Howard, *Dynamic Programming and Markov Processes*, 1960 (policy iteration).
* M. Falcone, R. Ferretti, *Semi-Lagrangian Approximation Schemes for Linear and
  Hamilton–Jacobi–Bellman Equations*, SIAM, 2013.
* D. P. Bertsekas, *Dynamic Programming and Optimal Control*, Athena Scientific.
* H. J. Kushner, P. Dupuis, *Numerical Methods for Stochastic Control Problems in
  Continuous Time* (Markov chain approximation), Springer, 2001.
