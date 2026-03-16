import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), p_drop=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, 1)] 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1) 