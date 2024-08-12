import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class mlp(nn.Module):
    def __init__(self, input_sizes):
        super(mlp, self).__init__()
        
        self.layers = nn.ModuleList()
        for i, o in zip(input_sizes[:-1], input_sizes[1:]):
            layer = nn.Linear(i, o)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.layers.append(layer)
    
    def forward(self, x):
        x = x[:, 1, :]
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))
        x = torch.tanh(self.layers[-1](x)) 
        return x


