import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import os

def standardize(x, epsilon=1e-6):
    # Compute mean and standard deviation
    mean = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    #print(mean)
    #print(std)

    # Add epsilon to standard deviation to prevent division by zero
    std = std + epsilon

    # Standardize
    return (x - mean) / std

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

class mlp(nn.Module):
    def __init__(self, input_sizes):
        super(mlp, self).__init__()
        
        # Attention mechanism paramaters
        self.att = nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3,3), nn.Tanh(), nn.Linear(3,1), nn.Tanh() ) #  nn.Softmax(dim=-1)  

        for layer in self.att:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 1)
                nn.init.zeros_(layer.bias)


        # Transformation u-values        
        self.layers = nn.ModuleList()
        for i, o in zip(input_sizes[:-1], input_sizes[1:]):
            layer = nn.Linear(i, o)
            init.xavier_normal_(layer.weight)
            init.zeros_(layer.bias)
            self.layers.append(layer)

        
    
    def forward(self, x):
        t_vals = x[:, 0, :] * 100
        u_vals = x[:, 1, :]

        
        
        t_vals_reshaped = t_vals.reshape(-1, 1)  # size (batch_size * 34, 1)
        
        att_params = self.att(t_vals_reshaped)
        
        # Reshape att_params back to (batch_size, 34)
        att_params = att_params.reshape(x.size(0), -1)  # size (batch_size, 34)

        x = u_vals * att_params # multiplied one by one
        
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))
        x = torch.tanh(self.layers[-1](x))
        return x

    
    def plot_attention(self, path, num):
        os.makedirs(f"figures/{path}/attentions", exist_ok=True)
        t_values_tensor = torch.tensor(t_vals).float().unsqueeze(-1) * 100.0 # Convert to tensor and add dimension
        with torch.no_grad():  # No need to track gradients for this
            att_params = self.att(t_values_tensor).squeeze(-1)  # Pass through att_fc and squeeze

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, att_params.numpy(), label='Attention Weights')
        plt.xlabel('Time Distance Values')
        plt.ylabel('Attention Weights')
        plt.title('Attention Weights over Time Distance Values')
        plt.legend()
        plt.savefig(f'figures/{path}/attentions/attention_{num}.png')
        plt.close()

        # Use t_vals information as a sort of filter / attention layer for u_values. I want to train parameters that tell me, 
        # given a vector u_vals of all 1s, how much the distance of these u_vals in time w.r.t. the output influences their importance.
        # It will be something like a learnt filter. This filter should not depend on u_vals as t is the real independent variable.

    
    
    


