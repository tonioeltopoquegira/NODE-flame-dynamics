import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class mlp(nn.Module):
    def __init__(self, input_branch_sizes, time_branch_sizes):
        super(mlp, self).__init__()
        
        # Create input branch
        self.input_layers = nn.ModuleList()
        for i, o in zip(input_branch_sizes[:-1], input_branch_sizes[1:]):
            layer = nn.Linear(i, o)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.input_layers.append(layer)
        
        # Create time branch
        self.time_layers = nn.ModuleList()
        for i, o in zip(time_branch_sizes[:-1], time_branch_sizes[1:]):
            layer = nn.Linear(i, o)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.time_layers.append(layer)
    
    def forward(self, x):
        # Process through time branch
        time_out = x[:, 0,:]
        for layer in self.time_layers[:-1]:
            time_out = F.tanh(layer(time_out))
        time_out = torch.tanh(self.time_layers[-1](time_out))
        
        # Process through input branch
        input_out = x[:, 1, :]
        for layer in self.input_layers[:-1]:
            input_out = F.tanh(layer(input_out))
        input_out = torch.tanh(self.input_layers[-1](input_out))
        
        # Compute dot product between the outputs of both branches
        dot_product = torch.sum(time_out * input_out, dim=-1, keepdim=True)
        
        return dot_product
