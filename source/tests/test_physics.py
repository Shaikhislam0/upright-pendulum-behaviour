from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from upright_pendulum.physics import (
    Parameters, dynamics, energy, horizontal_momentum, integrate, power,
)


def reference_rhs(t, state, force, p):
    """Independent matrix solve, rather than the production closed-form inverse."""
    _, v, theta, omega = state
    coupling = p.pole_mass * p.com_distance * np.cos(theta)
    matrix = np.array([[p.cart_mass + p.pole_mass, coupling],
                       [coupling, p.inertia_pivot]])
    rhs = [force - p.cart_damping * v + p.pole_mass * p.com_distance * np.sin(theta) * omega**2,
           p.pole_mass * p.gravity * p.com_distance * np.sin(theta) - p.pivot_damping * omega]
    a, alpha = np.linalg.solve(matrix, rhs)
    return np.array([v, a, omega, alpha])


def test_uniform_rod_parameters_and_energy_barrier():
    p = Parameters()
    assert p.com_distance == 0.5
    assert p.inertia_pivot == pytest.approx(0.1 / 3)
    assert energy([0, 0, np.pi, 0], p) == pytest.approx(0)
    assert energy([0, 0, 0, 0], p) == pytest.approx(0.981)


@pytest.mark.parametrize("theta", [0.0, np.pi, -np.pi, 2 * np.pi])
def test_equilibria(theta):
    np.testing.assert_allclose(dynamics([0, 0, theta, 0], 0), 0, atol=1e-14)


def test_force_and_gravity_signs():
    derivative = dynamics([0, 0, 0, 0], 1)
    assert derivative[1] > 0 and derivative[3] < 0
    assert dynamics([0, 0, 0.01, 0], 0)[3] > 0  # upright is unstable
    assert dynamics([0, 0, np.pi + 0.01, 0], 0)[3] < 0  # down is stable


def test_random_matrix_equations_symmetry_and_power_balance():
    rng = np.random.default_rng(143)
    p = Parameters(cart_damping=0.13, pivot_damping=0.017)
    for _ in range(100):
        state = rng.uniform([-2, -3, -10, -8], [2, 3, 10, 8])
        force = rng.uniform(-10, 10)
        rhs = dynamics(state, force, p)
        np.testing.assert_allclose(rhs, reference_rhs(0, state, force, p), rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(dynamics(-state, -force, p), -rhs, atol=1e-13)
        # Differentiate the independently implemented energy numerically.
        eps = 1e-5
        gradient = np.array([(energy(state + eps * axis, p) - energy(state - eps * axis, p))
                             / (2 * eps) for axis in np.eye(4)])
        assert gradient @ rhs == pytest.approx(power(state, force, p), abs=2e-7)


@pytest.mark.parametrize("initial", [
    [0, 0, 2.4, 0], [0, 0, np.pi, 10], [0.2, 1.5, 0.2, -2],
])
def test_unforced_energy_and_horizontal_momentum(initial):
    p = Parameters()
    state = np.array(initial, dtype=float)
    e0, momentum0 = energy(state, p), horizontal_momentum(state, p)
    for _ in range(500):  # 10 seconds; all control-step endpoints
        state = integrate(state, 0, 0.02, p)
        assert abs(energy(state, p) - e0) < 1e-6
        assert abs(horizontal_momentum(state, p) - momentum0) < 1e-7


def test_fourth_order_convergence_and_independent_reference():
    p = Parameters()
    initial = [0, 0, 2.4, 1.3]
    ref = solve_ivp(reference_rhs, [0, 2], initial, args=(1.7, p), method="DOP853",
                    rtol=1e-12, atol=1e-14)
    assert ref.success
    errors = [np.linalg.norm(integrate(initial, 1.7, 2, p, max_step=h) - ref.y[:, -1])
              for h in (0.02, 0.01, 0.005)]
    assert 10 < errors[0] / errors[1] < 22
    assert 10 < errors[1] / errors[2] < 22
    np.testing.assert_allclose(integrate(initial, 1.7, 2, p), ref.y[:, -1], atol=1e-7, rtol=0)


def test_downward_small_angle_frequency():
    p = Parameters()
    state = np.array([0, 0, np.pi, 0])
    eps = 1e-6
    jac = np.column_stack([(dynamics(state + eps * a, 0, p) - dynamics(state - eps * a, 0, p))
                           / (2 * eps) for a in np.eye(4)])
    frequency = np.sqrt((p.cart_mass + p.pole_mass) * p.pole_mass * p.gravity * p.com_distance
                        / ((p.cart_mass + p.pole_mass) * p.inertia_pivot
                           - (p.pole_mass * p.com_distance)**2))
    assert max(abs(np.linalg.eigvals(jac).imag)) == pytest.approx(frequency, rel=1e-8)


def test_damping_loses_energy_and_constant_force_momentum_impulse():
    initial = [0, 0.4, 2.2, 3]
    p = Parameters(cart_damping=0.1, pivot_damping=0.01)
    state = np.array(initial, dtype=float)
    for _ in range(100):
        previous_energy = energy(state, p)
        state = integrate(state, 0, 0.02, p)
        assert energy(state, p) <= previous_energy + 1e-12
    undamped = replace(p, cart_damping=0, pivot_damping=0)
    final = integrate(initial, 1.7, 2, undamped)
    assert horizontal_momentum(final, undamped) - horizontal_momentum(initial, undamped) == pytest.approx(3.4, abs=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"cart_mass": 0}, {"pole_mass": -1}, {"length": float("nan")},
    {"com_distance": 2}, {"inertia_com": -1}, {"cart_damping": -1},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_point_mass_limit():
    p = Parameters(com_distance=1, inertia_com=0)
    assert p.inertia_pivot == 0.1
    assert np.all(np.isfinite(dynamics([0, 0, 0, 0], 1, p)))
