import math
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Linear(i_dim, 128),
             nn.ReLU(inplace=True),
             nn.Linear(128, 128),
             nn.ReLU(inplace=True),
             nn.Linear(128, 128),
             nn.ReLU(inplace=True),
             nn.Linear(128, 128),
             nn.ReLU(inplace=True),
             nn.Linear(128, 1)]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, z_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        x = [nn.Linear(z_dim, 128),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Linear(128, 128),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Linear(128, 128),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Linear(128, o_dim)]

        self.x = nn.Sequential(*x)

    def forward(self, z):
        return self.x(z)

