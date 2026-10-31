"""SI units; state [x, x_dot, theta, theta_dot], theta clockwise from up.

The integrator uses float64 and a force held constant over a control interval.
The energy zero is the hanging-down configuration.
"""

from dataclasses import dataclass
from collections.abc import Callable
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Parameters:
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    length: float = 1.0
    gravity: float = 9.81
    com_distance: float | None = None
    inertia_com: float | None = None
    cart_damping: float = 0.0  # N s / m
    pivot_damping: float = 0.0  # N m s / rad

    def __post_init__(self):
        if self.com_distance is None:
            object.__setattr__(self, "com_distance", self.length / 2)
        if self.inertia_com is None:
            object.__setattr__(self, "inertia_com", self.pole_mass * self.length**2 / 12)
        for name in ("cart_mass", "pole_mass", "length", "gravity", "com_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("inertia_com", "cart_damping", "pivot_damping"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.com_distance > self.length:
            raise ValueError("com_distance cannot exceed total length")

    @property
    def inertia_pivot(self) -> float:
        return self.inertia_com + self.pole_mass * self.com_distance**2


def state_vector(state: ArrayLike) -> NDArray[np.float64]:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain four finite values")
    return state.copy()


def _rhs(state, force, p):
    _, velocity, theta, omega = state
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    total_mass = p.cart_mass + p.pole_mass
    coupling = p.pole_mass * p.com_distance * cos_theta
    inertia = p.inertia_pivot
    rhs_x = force - p.cart_damping * velocity + p.pole_mass * p.com_distance * sin_theta * omega**2
    rhs_theta = p.pole_mass * p.gravity * p.com_distance * sin_theta - p.pivot_damping * omega
    determinant = total_mass * inertia - coupling**2
    acceleration = (inertia * rhs_x - coupling * rhs_theta) / determinant
    angular_acceleration = (total_mass * rhs_theta - coupling * rhs_x) / determinant
    return np.array([velocity, acceleration, omega, angular_acceleration], dtype=np.float64)


def dynamics(state: ArrayLike, force: float, p: Parameters = Parameters()) -> NDArray[np.float64]:
    """Continuous nonlinear ODE. Force limiting belongs to the actuator/env."""
    state = state_vector(state)
    if not math.isfinite(force):
        raise ValueError("force must be finite")
    return _rhs(state, float(force), p)


def energy(state: ArrayLike, p: Parameters = Parameters()) -> float:
    """Total mechanical energy in joules; potential zero at theta = pi."""
    _, velocity, theta, omega = state_vector(state)
    kinetic = (
        0.5 * (p.cart_mass + p.pole_mass) * velocity**2
        + p.pole_mass * p.com_distance * math.cos(theta) * velocity * omega
        + 0.5 * p.inertia_pivot * omega**2
    )
    potential = p.pole_mass * p.gravity * p.com_distance * (1 + math.cos(theta))
    return float(kinetic + potential)


def horizontal_momentum(state: ArrayLike, p: Parameters = Parameters()) -> float:
    _, velocity, theta, omega = state_vector(state)
    return float((p.cart_mass + p.pole_mass) * velocity
                 + p.pole_mass * p.com_distance * math.cos(theta) * omega)


def power(state: ArrayLike, force: float, p: Parameters = Parameters()) -> float:
    """dE/dt = applied power minus viscous dissipation (watts)."""
    _, velocity, _, omega = state_vector(state)
    if not math.isfinite(force):
        raise ValueError("force must be finite")
    return float(force * velocity - p.cart_damping * velocity**2 - p.pivot_damping * omega**2)


def integrate(
    state: ArrayLike, force: float, duration: float,
    p: Parameters = Parameters(), *, max_step: float = 0.002,
    observer: Callable[[float, NDArray[np.float64]], None] | None = None,
) -> NDArray[np.float64]:
    """Fixed-substep classical RK4, with equal subdivisions of duration.

    RK4 is not exactly energy preserving; convergence and finite-time drift
    must be validated over the speeds and durations of the experiment.
    An optional observer receives each completed substep's relative time and
    a copy of its state. Observing cannot mutate the integrated trajectory.
    """
    state = state_vector(state)
    if not math.isfinite(force):
        raise ValueError("force must be finite")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable or None")
    n = math.ceil(duration / max_step)
    h = duration / n
    with np.errstate(over="raise", invalid="raise"):
        for index in range(n):
            k1 = _rhs(state, force, p)
            k2 = _rhs(state + h * k1 / 2, force, p)
            k3 = _rhs(state + h * k2 / 2, force, p)
            k4 = _rhs(state + h * k3, force, p)
            state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if observer is not None:
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError("integration produced a nonfinite state")
                offset = duration if index == n - 1 else (index + 1) * h
                observer(offset, state.copy())
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("integration produced a nonfinite state")
    return state
