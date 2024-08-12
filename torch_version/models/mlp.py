import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules.utils import choose_nonlinearity

class mlp(nn.Module):
    def __init__(self, input_sizes, nonlinearity, time_dependency, time_sizes):
        super(mlp, self).__init__()

        self.time_dependency = time_dependency

        if input_sizes[-1] != 1:
                raise ValueError("Error in the sizes of input branch. should have same: input_sizes[-1] == 1")

        # Create time branch on the style of DeepONets
        if time_dependency == "time-branch":
            if input_sizes[0] != time_sizes[0] or time_sizes[-1] != 1:
                raise ValueError("Error in the sizes of time branch: should have same: (1) input_sizes[0] == time_sizes[0] (2) time_sizes[-1] == 1")
            self.time_layers = nn.ModuleList()
            for i, o in zip(time_sizes[:-1], time_sizes[1:]):
                layer = nn.Linear(i, o)
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                self.time_layers.append(layer)

        # Apply a sort of time-attention mechanism
        if time_dependency == "time-att":
            if time_sizes[0] != 1 or time_sizes[-1] != 1:
                raise ValueError("Error in the sizes of time attention: should have same: (1) time_sizes[0] == 1 (2) time_sizes[-1] == 1")
            self.time_layers = nn.ModuleList()
            for i, o in zip(time_sizes[:-1], time_sizes[1:]):
                layer = nn.Linear(i, o)
                nn.init.normal_(layer.weight, 0.0, 1)
                nn.init.zeros_(layer.bias)
                self.time_layers.append(layer)
        
        # Define Layers from input_sizes
        self.layers = nn.ModuleList()
        for i, o in zip(input_sizes[:-1], input_sizes[1:]):
            layer = nn.Linear(i, o)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.layers.append(layer)
        
        # Nonlinearity of choice
        self.nonlinearity = choose_nonlinearity(nonlinearity)
    
    def forward(self, x):
        time = x[:, 0,:]
        x = x[:, 1, :]
        
        if self.time_dependency == "time-att":

            # Reshape to a row vector and adjust norm
            t_vals = time * 100
            t_vals_reshaped = t_vals.reshape(-1, 1)

            # Apply "attention" non linearity to linear time
            for layer in self.time_layers[:-1]:
                t_vals_reshaped = self.nonlinearity(layer(t_vals_reshaped))
            att_params = self.nonlinearity(self.time_layers[-1](t_vals_reshaped))
            att_params = att_params.reshape(x.size(0), -1)

            # Filter the values of u based on these parameters
            x = x * att_params 
        
        # Apply transformation to u values
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))
        out = self.nonlinearity(self.layers[-1](x))


        if self.time_dependency == "time-branch":
            # Process through time branch
            for layer in self.time_layers[:-1]:
                time = self.nonlinearity(layer(time))
            time = self.nonlinearity(self.time_layers[-1](time))

            # Dot product with u values result
            out = torch.sum(time * out, dim=-1, keepdim=True)
        
        return out
