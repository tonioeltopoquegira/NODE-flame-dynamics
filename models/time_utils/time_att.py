import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp


class att_mlp(nn.Module):

    time_sizes: list

    def setup(self):

        # Initializers    
        dense_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.zeros_init()

        self.time_layers = [nn.Dense(o, kernel_init=dense_init, bias_init=bias_init) for o in self.time_sizes[1:]]

    def __call__(self, time):
        
        # Reshape to a row vector and adjust norm
        t_vals = time * 100
        t_vals_reshaped = t_vals.reshape(-1, 1)

        # Apply "attention" non-linearity to linear time
        for layer in self.time_layers[:-1]:
            t_vals_reshaped = nn.relu(layer(t_vals_reshaped))
        att_params = nn.relu(self.time_layers[-1](t_vals_reshaped))
        
        #att_params = att_params.reshape(x.shape[0], -1)
        
        return att_params

    def plot_attention(self, path, num):

        os.makedirs(f"figures/{path}/attentions", exist_ok=True)

        t_vals = [ 9.9000e-03,  9.8000e-03,  9.7000e-03,  9.6000e-03,  9.5000e-03,
          9.4000e-03,  9.3000e-03,  9.2000e-03,  9.1000e-03,  9.0000e-03,
          8.9000e-03,  8.8000e-03,  8.7000e-03,  8.6000e-03,  8.5000e-03,
          8.4000e-03,  8.3000e-03,  8.2000e-03,  8.1000e-03,  8.0000e-03,
          7.9000e-03,  7.8000e-03,  7.7000e-03,  7.6000e-03,  7.5000e-03,
          7.4000e-03,  7.3000e-03,  7.2000e-03,  7.1000e-03,  7.0000e-03,
          6.9000e-03,  6.8000e-03,  6.7000e-03,  6.6000e-03,  6.5000e-03,
          6.4000e-03,  6.3000e-03,  6.2000e-03,  6.1000e-03,  6.0000e-03,
          5.9000e-03,  5.8000e-03,  5.7000e-03,  5.6000e-03,  5.5000e-03,
          5.4000e-03,  5.3000e-03,  5.2000e-03,  5.1000e-03,  5.0000e-03,
          4.9000e-03,  4.8000e-03,  4.7000e-03,  4.6000e-03,  4.5000e-03,
          4.4000e-03,  4.3000e-03,  4.2000e-03,  4.1000e-03,  4.0000e-03,
          3.9000e-03,  3.8000e-03,  3.7000e-03,  3.6000e-03,  3.5000e-03,
          3.4000e-03,  3.3000e-03,  3.2000e-03,  3.1000e-03,  3.0000e-03,
          2.9000e-03,  2.8000e-03,  2.7000e-03,  2.6000e-03,  2.5000e-03,
          2.4000e-03,  2.3000e-03,  2.2000e-03,  2.1000e-03,  2.0000e-03,
          1.9000e-03,  1.8000e-03,  1.7000e-03,  1.6000e-03,  1.5000e-03,
          1.4000e-03,  1.3000e-03,  1.2000e-03,  1.1000e-03,  1.0000e-03,
          9.0000e-04,  8.0000e-04,  7.0000e-04,  6.0000e-04,  5.0000e-04,
          4.0000e-04,  3.0000e-04,  2.0000e-04,  1.0000e-04, -0.0000e+00]
        
        t_values = jnp.array(t_vals) * 100.0 # Convert to tensor and add dimension
        
        att_params = self(t_values)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, att_params, label='Attention Weights')
        plt.xlabel('Time Distance Values')
        plt.ylabel('Attention Weights')
        plt.title('Attention Weights over Time Distance Values')
        plt.legend()
        plt.savefig(f'figures/{path}/attentions/attention_{num}.png')
        plt.close()
