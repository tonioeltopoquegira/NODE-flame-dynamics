import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class mlp(nn.Module):
    def __init__(self, input):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input, 10)
        self.fc2 = nn.Linear(10, 1)
        """self.fc_y1 = nn.Linear(1, 3)
        self.fc_y2 = nn.Linear(3, 1)"""
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        #nn.init.zeros_(self.fc1.weight)
        #nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        """nn.init.zeros_(self.fc_y1.weight)
        nn.init.zeros_(self.fc_y2.weight)
        nn.init.zeros_(self.fc_y1.bias)
        nn.init.zeros_(self.fc_y2.bias)"""


        
    def forward(self, t0, y0, u):
        """
        Args: 
            t0: torch.Tensor 0-D time at which we evaluate the derivative
            y0: torch.Tensor 0-D previous output point
            u: torch.Tensor 0-D other input 
        Returns:
            dqdt: derivative at that point
        """

        x = F.relu(self.fc1(u))
        x = F.tanh(self.fc2(x))
        return x

        