from .solvers import FixedGridODESolver
import numpy as np


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0, x):
        
        f0 = func(t0, y0, x)
    
        return dt * f0 * 100
    

# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t0, dt, t1, y0, x):
    
    k1 = dt * func(t0, y0, x)
    k2 = dt * func(t0 + dt / 2, y0 + k1/2,  x + k1 / 2 * np.ones_like(x))
    k3 = dt * func(t0 + dt / 2, y0 + k2/2, x + k2 / 2 * np.ones_like(x))
    k4 = dt * func(t0 + dt,y0 +k3/2, x + k3 * np.ones_like(x))
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t0, dt, t1, y0, x):
    """Smaller error with slightly more compute."""

    k1 = func(t0, y0, x)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third, x)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third), x)
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), x)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125
    

class RK4(FixedGridODESolver):
    order = 4
    def _step_func(self, func, t0, dt, t1, y0, x):
        return rk4_step_func(func, t0, dt, t1, y0, x)