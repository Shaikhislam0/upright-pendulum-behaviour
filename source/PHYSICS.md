# Physical simulation: derivation and validation

Extracted from the validated Stage-1 documentation. The base environment returns raw state and zero reward; PPO adds explicit observation, reward and action wrappers.

## Physical assumptions and conventions

All quantities use SI units. The cart moves on an unbounded, horizontal rail;
its pivot height is fixed. The pendulum is a planar, rigid **uniform slender rod**
by default. The pivot is frictionless and the cart has no drag by default.
There are no collisions, actuator lag, or hidden forces.

Mass and length alone do not fix a rigid body's mass distribution. Uniformity is
an explicit provisional assumption. The center-of-mass distance and moment of
inertia can be set independently in `Parameters` to represent another body.
If either is omitted, its uniform-rod value is used; when changing the mass
distribution, specify **both** explicitly. A point mass at the tip is represented
by `com_distance=length, inertia_com=0`.

| Symbol | Meaning | Default |
|---|---|---|
| M | Cart mass | 1.0 kg |
| m | Pendulum mass | 0.1 kg |
| L | Total rod length | 1.0 m |
| l | Pivot to center of mass | L/2 = 0.5 m |
| I_c | Inertia about center of mass | mL²/12 = 0.008333… kg m² |
| J | Inertia about pivot | I_c + ml² = mL²/3 = 0.033333… kg m² |
| g | Gravitational acceleration | 9.81 m/s² |
| b | Cart viscous damping | 0 N s/m |
| c | Pivot viscous damping | 0 N m s/rad |
| F | Horizontal control force, positive right | [−10, +10] N |

The state order is `[x, x_dot, theta, theta_dot]`. Positive x points right,
positive y points up, and positive theta rotates **clockwise from upward vertical**
in this plane. Thus the center of mass is at

\[
X_c=x+l\sin\theta,\qquad Y_c=l\cos\theta.
\]

The initial state is exactly `[0, 0, pi, 0]`. Upright equilibria are
`[x, 0, 2k*pi, 0]`; the desired one has x=0. The passive system has no mechanism
that centers the cart. The downward state is an equilibrium too: without force
or a perturbation it remains down, to numerical precision.

Angles are **unwrapped** in state and observations: full rotations remain
continuous. A wrapped angle error in [−pi, pi] is provided only as an `info`
diagnostic. No angle threshold ends an episode.

## Nonlinear derivation

Differentiating the center-of-mass coordinates gives

\[
\dot X_c=\dot x+l\cos\theta\dot\theta,\qquad
\dot Y_c=-l\sin\theta\dot\theta.
\]

Adding cart translation, rod center-of-mass translation, and rod rotation gives
the kinetic and potential energies

\[
K=\tfrac12(M+m)\dot x^2+ml\cos\theta\dot x\dot\theta
  +\tfrac12J\dot\theta^2,\qquad
V=mgl(1+\cos\theta).
\]

The additive constant sets potential energy to zero at the bottom. With
Lagrangian \(\mathcal L=K-V\) and generalized forces
\(Q_x=F-b\dot x\), \(Q_\theta=-c\dot\theta\), the Euler–Lagrange equations yield

\[
(M+m)\ddot x+ml\cos\theta\ddot\theta
 -ml\sin\theta\dot\theta^2=F-b\dot x,
\]
\[
ml\cos\theta\ddot x+J\ddot\theta-mgl\sin\theta=-c\dot\theta.
\]

Define \(A=M+m\), \(B=ml\cos\theta\),
\(r_x=F-b\dot x+ml\sin\theta\dot\theta^2\), and
\(r_\theta=mgl\sin\theta-c\dot\theta\). Then

\[
\begin{bmatrix}A&B\\B&J\end{bmatrix}
\begin{bmatrix}\ddot x\\\ddot\theta\end{bmatrix}
=\begin{bmatrix}r_x\\r_\theta\end{bmatrix},\qquad
D=AJ-B^2>0,
\]
\[
\boxed{\ddot x=\frac{Jr_x-Br_\theta}{D}},\qquad
\boxed{\ddot\theta=\frac{Ar_\theta-Br_x}{D}}.
\]

For positive masses and l>0, \(D\ge (M+m)I_c+Mm l^2>0\).
These are fully nonlinear equations; no small-angle approximation is used in
the simulator. As a sign check, a positive force at upright accelerates the
cart right and the rod counterclockwise (negative angular acceleration).

The total mechanical energy and horizontal momentum obey

\[
E=K+V,\qquad \dot E=F\dot x-b\dot x^2-c\dot\theta^2,
\]
\[
p_x=(M+m)\dot x+ml\cos\theta\dot\theta,\qquad
\dot p_x=F-b\dot x.
\]

Therefore E and p_x are conserved when F=b=c=0. The energy difference from
stationary downward to stationary upright is 2mgl = **0.981 J**. Energy
conservation alone is insufficient to establish correct dynamics, which is why
sign, momentum, power-balance, and independent integration checks are included.

The derivation follows the standard Lagrangian/manipulator formulation in
[MIT Underactuated Robotics](https://underactuated.mit.edu/multibody.html).
The rod inertia and upward-angle convention are specified here explicitly.

## Numerical and environment decisions

- Double precision throughout the physics and observations.
- Control period: 0.02 s (50 Hz), with force held constant during each interval.
- Classical fourth-order Runge–Kutta, subdividing each interval into equal steps
  no longer than 0.002 s (500 Hz by default). No energy projection or artificial
  damping masks integration errors.
- RK4 is not symplectic and does not conserve energy exactly. The acceptance
  criterion is finite-time numerical drift plus step-halving convergence;
  longer runs, faster rotations, or changed parameters require revalidation.
- Actions have shape `(1,)` and express force directly in newtons. Nonfinite,
  malformed, and out-of-range actions are rejected rather than silently clipped.
- Observations have shape `(4,)`, dtype float64, and unbounded limits. There
  are no rail boundaries until a physical track length is specified.
- Reward is exactly 0.0 as a Stage-1 placeholder. `terminated` is always false.
  The default 1,000-step (20 s) limit sets `truncated`; it is a provisional
  evaluation horizon, not a physical failure or success condition.
- `reset(seed=...)` follows Gymnasium's RNG contract. Default reset is exact and
  deterministic. Optional `noise_scale` adds independently sampled uniform
  noise with explicit per-component half-widths, for controlled tests only.
- Optional ANSI rendering prints the state. No graphics dependencies are
  required to run the environment.

The API and termination semantics follow the official
[Gymnasium Env documentation](https://gymnasium.farama.org/api/env/).
For a future PPO policy, a periodic observation wrapper such as sin/cos(theta)
can be evaluated separately while retaining this physical state internally.

## Install and reproduce (PowerShell)

Using a normal Python 3.11+ installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[validation]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m upright_pendulum.validate --output validation
```

An existing local `.venv` was used for the supplied validation results.
`requirements-lock.txt` records the original physics/validation dependency
versions. `requirements-training-lock.txt` records the expanded, tested PPO
environment and includes the CPU PyTorch wheel index. The results also record
the Python and principal library versions.
To recreate those dependencies, install from that file, then install this
package with `pip install --no-deps --no-build-isolation -e .`.

```python
import numpy as np
from upright_pendulum import CartPendulumEnv

env = CartPendulumEnv(render_mode="ansi")
obs, info = env.reset(seed=2026)
obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
print(env.render())
env.close()
```

The environment is also registered as `UprightPendulum/CartPendulum-v0` after
importing `upright_pendulum`. For physics experiments without a time limit,
construct `CartPendulumEnv(max_episode_steps=None)` directly. Gymnasium reserves
`gym.make(..., max_episode_steps=...)` for its own TimeLimit wrapper; use the
direct constructor when configuring this environment's internal horizon.

## Validation protocol and outputs

`tests/` checks the Gymnasium contract, seeding, observation ownership, action
validation, time-limit semantics, uninterrupted rotations, physical equilibrium
and force signs, mass-matrix equations, reflection symmetry, infinitesimal
power balance, horizontal impulse, damping, and the analytical small-oscillation
frequency around the hanging equilibrium.

`python -m upright_pendulum.validate` writes:

- `validation/report.json`: parameters, versions, predefined tolerances, measured
  errors, and an overall pass/fail flag. A failed check exits with an error.
- Four CSV trajectories at control-step endpoints: oscillation, rotation,
  coupled cart/rod motion, and release near upright, each **60 s** with F=b=c=0.
- `validation/physics_validation.png` and `.pdf`: energy drift, momentum drift,
  and timestep-convergence diagnostics.

Acceptance criteria, fixed in the validation code before running:

1. Maximum sampled energy drift < 1e−6 J for every unforced 60 s trajectory.
2. Maximum sampled momentum drift < 1e−7 kg m/s for every such trajectory.
3. Maximum error of each state component versus DOP853 < 1e−7 in its SI unit,
   sampled every 0.02 s over a 2 s forced trajectory (reference tolerances
   rtol=1e−12, atol=1e−14).
4. Final-state errors decrease by a factor between 10 and 22 on each halving
   0.02 → 0.01 → 0.005 s, consistent with fourth order (asymptotic factor 16).
5. Constant-force work and damped-unforced energy balance residuals < 1e−6 J
   over separate 5 s runs; damping losses are integrated using composite Simpson.

The reference solver uses a separate matrix solve for accelerations. It checks
the explicit acceleration formulas and integration, but shares the derived
physical model; it is not experimental validation of a real apparatus.
The energy drift is scaled by max(|E(0)|, 2mgl), avoiding an ill-defined relative
error when energy starts at zero. Unforced errors are sampled at control-step
endpoints, not all internal RK4 stages.

Stage 1 has passed its numerical checks and was independently reproduced by
the user. Its physical assumptions still define the scope of validity.
Reward and observation options have since been added as separate wrappers.
The current generation/cohort training protocol is in the root README and
EXPERIMENT_PROTOCOL.md; these numerical results concern the physical simulator.

## Measured Stage-1 results

The supplied run passed all 36 tests and every numerical acceptance criterion.
The Gymnasium checker emits advisory warnings for the intentional physical
force bounds and unbounded state space.

| 60 s unforced trajectory | Maximum energy drift (J) | Maximum momentum drift (kg m/s) |
|---|---:|---:|
| Oscillation | 1.18e−11 | 3.29e−12 |
| Rotation | 3.94e−10 | 3.87e−10 |
| Coupled cart/rod motion | 5.53e−11 | 8.06e−11 |
| Release near upright | 5.86e−11 | 7.75e−11 |

Step-halving error ratios were 16.01 and 15.87. Over the forced 2 s reference
trajectory, maximum component errors were 7.59e−12 m, 2.31e−11 m/s,
1.66e−10 rad, and 4.63e−10 rad/s. Constant-force work and damped energy-balance
residual magnitudes were 1.81e−11 J and 4.10e−11 J respectively.
These results validate the tested numerical regimes, subject to the stated
physical assumptions and finite-time limits.

