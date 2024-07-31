import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Tanh()
        )
        
    def forward(self, t0, y0, u):
        """
        Args: 
            t0: torch.Tensor 0-D time at which we evaluate the derivative
            y0: torch.Tensor 0-D previous output point
            u: torch.Tensor 0-D other input 
        Returns:
            dqdt: derivative at that point
        """

        

        t0 = t0.unsqueeze(0)  
        y0 = y0.reshape(1)

        

        ins = torch.cat([t0, y0, u], dim=0)

        out = self.seq(ins)

        return out