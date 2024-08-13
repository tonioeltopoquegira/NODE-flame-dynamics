import flax.linen as nn
import jax.numpy as jnp
from models.node_utils.integrator import Integrator

# NOTE!
# (1) Interpolator for two dimensions of x are different (one should just be the transf of times --> distance time, the other a proper interpolator)
# (2) How do you choose w.r.t. which values of amplitude to interpolate?


class NeuralODE(nn.Module):

    f_prime_model: nn.module
    integ_strat: str
    integ_meth: str 
    interp: nn.module

    def setup(self):

        self.integrator = Integrator(strategy=self.integ_strat, method=self.integ_meth, interp=self.interp)

    def __call__(self, times, x, iv):
    
        t_evaluation, out = self.integrator(self.f_prime_model, t_evaluation=times, y0=iv, other_inputs = x)

        return t_evaluation, out