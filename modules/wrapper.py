import torch
import torch.nn as nn

from integrator.adjoint import odeint_adjoint as odeint
#from integrator.odeint import odeint

from integrator.misc import get_param_shapes


class NeuralODE(nn.Module):

    def __init__(self, f_prime_model, integrator):
        super(NeuralODE, self).__init__()

        self.f_prime_model = f_prime_model # this is a nn.Module that takes in its forward (t, Q0)
        self.integrator = integrator # this is a string code to choose integrator
        self.shapes = get_param_shapes(f_prime_model)
        # TODO Interpolator compatible with torch ?

    def forward(self, x, iv):

        # Get Times to evaluate
        eval_times = x[:, 1, :].squeeze(1).squeeze(0)

        # Get Initial value for Q
        initial_value = iv.squeeze()
       
        # Velocity perturbations
        u_values = x[:, 0, :].squeeze(1)

        # Gradients?
        u_values.requires_grad_(True)
        initial_value.requires_grad_(False)
        eval_times.requires_grad_(False)

        out = odeint(self.f_prime_model, y0=initial_value, t=eval_times, x=u_values, method=self.integrator, shapes = self.shapes)

        out = out.unsqueeze(0).unsqueeze(0)
        
        
        return out


