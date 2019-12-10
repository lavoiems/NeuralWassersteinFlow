import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Linear(i_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, 1)]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        x = [nn.Linear(o_dim+1, h_dim),
             nn.ReLU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ReLU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ReLU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ReLU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ReLU(inplace=True),
             nn.Linear(h_dim, o_dim)]

        self.x = nn.Sequential(*x)

    def forward(self, z, t):
        o = torch.cat((z, t), 1)
        return self.x(o)
