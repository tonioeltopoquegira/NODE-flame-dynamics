import torch
import torch.nn as nn

class NeuralODE(nn.Module):

    def __init__(self, f_prime_model, integrator):
        super(NeuralODE, self).__init__()

        self.f_prime_model = f_prime_model # this is a nn.Module that takes in its forward (t, Q0)
        self.integrator = integrator # this is a string code to choose integrator
        self.shapes = get_param_shapes(f_prime_model)
        # TODO Interpolator compatible with torch ?

    def forward(self, times, x, iv):

        # Get Times to evaluate
        eval_times = times.squeeze(0)

        # Get Initial value for Q
        initial_value = iv

        
        # Velocity perturbations
        u_values = x[:, 1, :, :].squeeze(1).squeeze(0)

        # Gradients? Not sure about this!
        u_values.requires_grad_(True)
        initial_value.requires_grad_(False)
        eval_times.requires_grad_(False)

        out = odeint(self.f_prime_model, y0=initial_value, t=eval_times, x=u_values, method=self.integrator, shapes = self.shapes)
        out = out.permute(1, 0)
        
        
        return out


