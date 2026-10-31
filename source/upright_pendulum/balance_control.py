"""Local reference feedback, independent of PPO and its reward.

This is a balancing diagnostic, not a swing-up controller. SciPy is optional
and is imported only when designing the sampled-data LQR gain.
"""

import math

import numpy as np

from .physics import Parameters, state_vector


def upright_linearization(parameters: Parameters = Parameters()):
    """Return continuous-time A, B about [0, 0, 0, 0], with force in N.

    Linearized acceleration equations are
    [M+m, ml; ml, J] [xdd, theta_dd] = [F-b*v, m*g*l*theta-c*omega].
    The positive clockwise angle convention is retained.
    """
    p = parameters
    ml = p.pole_mass * p.com_distance
    mass = np.array([[p.cart_mass + p.pole_mass, ml], [ml, p.inertia_pivot]])
    acceleration_state = np.linalg.solve(mass, np.array([
        [0, -p.cart_damping, 0, 0],
        [0, 0, ml * p.gravity, -p.pivot_damping],
    ]))
    acceleration_force = np.linalg.solve(mass, np.array([[1.0], [0.0]]))
    a = np.zeros((4, 4))
    a[0, 1] = a[2, 3] = 1
    a[[1, 3], :] = acceleration_state
    b = np.zeros((4, 1))
    b[[1, 3], :] = acceleration_force
    return a, b


def design_balance_lqr(parameters: Parameters = Parameters(), *, control_dt=0.02):
    """Exact ZOH linear discretization and discrete infinite-horizon LQR.

    Fixed diagnostic cost per sample: z.T Q z + R F**2, Q=diag(10,1,100,1),
    R=0.1, z=[x (m), v (m/s), wrapped theta (rad), omega (rad/s)].
    These dimensional design weights do not modify the RL reward.
    """
    from scipy.linalg import expm, solve_discrete_are

    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    a, b = upright_linearization(parameters)
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = a
    augmented[:4, 4:] = b
    discrete = expm(control_dt * augmented)
    ad, bd = discrete[:4, :4], discrete[:4, 4:]
    q, r = np.diag([10.0, 1.0, 100.0, 1.0]), np.array([[0.1]])
    value = solve_discrete_are(ad, bd, q, r)
    gain = np.linalg.solve(r + bd.T @ value @ bd, bd.T @ value @ ad)
    return {"A": a, "B": b, "Ad": ad, "Bd": bd, "Q": q, "R": r,
            "P": value, "K": gain}


def balance_force(state, gain, *, force_limit=10.0):
    """F=clip(-K z, +/- force_limit); only error angle is wrapped."""
    if not math.isfinite(force_limit) or force_limit <= 0:
        raise ValueError("force_limit must be finite and positive")
    error = state_vector(state)
    error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
    gain = np.asarray(gain, dtype=np.float64)
    if gain.shape != (1, 4) or not np.all(np.isfinite(gain)):
        raise ValueError("gain must be a finite 1-by-4 matrix")
    requested = -(gain @ error).item()
    return float(np.clip(requested, -force_limit, force_limit))
