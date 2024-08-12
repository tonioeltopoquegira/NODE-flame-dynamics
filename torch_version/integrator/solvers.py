import abc
import torch
from .misc import _handle_unused_kwargs

class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", **unused_kwargs):

        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp    


    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, x): #TODO change this to use RK--> values of x at intermediate points must be provided
        time_grid = t

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        
        y0 = self.y0
        for i, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
            dt = t1 - t0 # could make it fancier
            dy = self._step_func(self.func, t0, dt, t1, y0, x[i, :])
            solution[i+1] = y0 + dy
            y0 = solution[i+1]
        return solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
