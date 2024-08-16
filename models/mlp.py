import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp

from modules.utils import choose_nonlinearity

class mlp(nn.Module):
    input_sizes: list
    nonlinearity: str
    time_dependency: str
    time_sizes: list
    initializer: str

    def setup(self):

        # Initializers
        if self.initializer[0] == 1:
            dense_init = nn.initializers.xavier_uniform()

        elif self.initializer[0] == 2:
            dense_init = nn.initializers.normal(0.1)
        
        elif self.initializer[0] == 0:
            dense_init = nn.initializers.zeros_init()
        
        bias_init = nn.initializers.constant(self.initializer[1])

        # Ensure the input_sizes and time_sizes are correctly configured
        if self.input_sizes[-1] != 1:
            raise ValueError("Error in the sizes of input branch. Should have same: input_sizes[-1] == 1")

        if self.time_dependency == "time-branch":
            if self.input_sizes[0] != self.time_sizes[0] or self.time_sizes[-1] != 1:
                raise ValueError("Error in the sizes of time branch: should have same: (1) input_sizes[0] == time_sizes[0] (2) time_sizes[-1] == 1")
            self.time_layers = [nn.Dense(o, kernel_init=dense_init, bias_init=bias_init) for o in self.time_sizes[1:]]
        
        elif self.time_dependency == "time-att":
            if self.time_sizes[0] != 1 or self.time_sizes[-1] != 1:
                raise ValueError("Error in the sizes of time attention: should have same: (1) time_sizes[0] == 1 (2) time_sizes[-1] == 1")
            self.time_layers = [nn.Dense(o, kernel_init=dense_init, bias_init=bias_init) for o in self.time_sizes[1:]]
            # self.time_att = 
        
        # Define Layers from input_sizes
        self.layers = [nn.Dense(o, kernel_init=dense_init, bias_init=bias_init) for o in self.input_sizes[1:]]
        
        # Nonlinearity of choice
        self.nonlinearity_fn = choose_nonlinearity(self.nonlinearity)


    def __call__(self, x):
        
        time = x[:, 0, :]
        x = x[:, 1, :]

        if self.time_dependency == "time-att":

            # Reshape to a row vector and adjust norm
            t_vals = time * 100
            t_vals_reshaped = t_vals.reshape(-1, 1)

            # Apply "attention" non-linearity to linear time
            for layer in self.time_layers[:-1]:
                t_vals_reshaped = self.nonlinearity_fn(layer(t_vals_reshaped))
            att_params = self.nonlinearity_fn(self.time_layers[-1](t_vals_reshaped))

            
            # att_params = 
            att_params = att_params.reshape(x.shape[0], -1)

            # Filter the values of u based on these parameters
            x = x * att_params 
        
        # Apply transformation to u values
        for layer in self.layers[:-1]:
            x = self.nonlinearity_fn(layer(x))
        out = nn.tanh(self.layers[-1](x))

        if self.time_dependency == "time-branch":

            # Process through time branch
            for layer in self.time_layers[:-1]:
                time = self.nonlinearity_fn(layer(time))
            time = self.nonlinearity_fn(self.time_layers[-1](time))

            # Dot product with u values result
            out = jnp.sum(time * out, axis=-1, keepdims=True)
        
        return out

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
        for layer in self.time_layers[:-1]:
            t_values = self.nonlinearity_fn(layer(t_values))
        att_params = self.nonlinearity_fn(self.time_layers[-1](t_values))
        

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, att_params, label='Attention Weights')
        plt.xlabel('Time Distance Values')
        plt.ylabel('Attention Weights')
        plt.title('Attention Weights over Time Distance Values')
        plt.legend()
        plt.savefig(f'figures/{path}/attentions/attention_{num}.png')
        plt.close()
