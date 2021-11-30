import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden), 
            nn.BatchNorm1d(hidden), 
            nn.ReLU(inplace = True), 
            nn.Linear(hidden, out_channel)
        )

    def forward(self, x):
        return self.net(x)