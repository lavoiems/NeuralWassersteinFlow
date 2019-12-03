import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Linear(i_dim+1, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, h_dim),
             nn.ELU(),
             nn.Linear(h_dim, 1)]

        self.x = nn.Sequential(*x)

    def forward(self, x, t):
        return self.x(torch.cat((x, t), 1)).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, h_dim, z_dim, **kwargs):
        super(Generator, self).__init__()

        x = [nn.Linear(z_dim+1, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, h_dim),
             nn.ELU(inplace=True),
             nn.Linear(h_dim, o_dim)]

        self.x = nn.Sequential(*x)

    def forward(self, z, t):
        return self.x(torch.cat((z, t), 1))
