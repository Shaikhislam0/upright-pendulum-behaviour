"""Full-task feasibility reference: energy shaping, then sampled LQR.

This controller is not used as the PPO policy, training data, or reward. Angle
is clockwise from up, and the uniform-rod inertia is retained throughout.
"""

import math
import numpy as np

from .balance_control import balance_force, design_balance_lqr
from .physics import Parameters, state_vector


class EnergySwingupController:
    def __init__(self, *, parameters=None, kick_direction=1):
        if kick_direction not in (-1, 1):
            raise ValueError("kick_direction must be -1 or +1")
        self.parameters = parameters or Parameters()
        self.gain = design_balance_lqr(self.parameters)["K"]
        self.kick_direction = kick_direction
        self.steps = 0
        self.balancing = False

    def force(self, state):
        p = self.parameters
        x, v, theta, omega = state_vector(state)
        angle = math.atan2(math.sin(theta), math.cos(theta))
        if abs(angle) < .25 and abs(omega) < 2.5 and abs(x) < 3:
            self.balancing = True
        if abs(angle) > .6:
            self.balancing = False
        if self.steps < 10:
            # Exactly downward and stationary is a passive equilibrium.
            force = self.kick_direction * 2.0
        elif self.balancing:
            force = balance_force(state, self.gain)
        else:
            ml = p.pole_mass * p.com_distance
            cosine, sine = math.cos(theta), math.sin(theta)
            pendulum_energy_error = .5 * p.inertia_pivot * omega ** 2 + ml * p.gravity * (cosine - 1)
            desired_acceleration = 5.0 * pendulum_energy_error * omega * cosine - .5 * x - v
            angular_acceleration = (ml * p.gravity * sine - p.pivot_damping * omega
                                    - ml * cosine * desired_acceleration) / p.inertia_pivot
            force = ((p.cart_mass + p.pole_mass) * desired_acceleration + ml * cosine * angular_acceleration
                     - ml * sine * omega ** 2 + p.cart_damping * v)
        self.steps += 1
        return float(np.clip(force, -10., 10.))
