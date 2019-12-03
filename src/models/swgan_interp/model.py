import math
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True)]

        dim = h_dim
        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        for _ in range(n_layers):
            in_dim = dim
            dim *= 2
            x += [nn.Conv2d(in_dim, dim, 4, 2, 1),
                  nn.LeakyReLU(0.2, inplace=True)]
        x += [nn.Conv2d(dim, dim, 4, 1, 0)]
        self.x = nn.Sequential(*x)

        self.out = nn.Sequential(nn.Linear(dim+1, dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(dim, dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(dim, 1))

    def forward(self, x, t):
        o = self.x(x).squeeze()
        o = torch.cat((o, t), 1)
        return self.out(o).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, kernel_dim, z_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        dim = h_dim * n_layers**2
        decoder = [nn.Conv2d(z_dim+1, dim, 1, 1, 0),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(dim, dim, 4, 1, 0),
                   nn.ReLU(True)]

        for _ in range(n_layers):
            in_dim = dim
            dim //= 2
            decoder += [nn.ConvTranspose2d(in_dim, dim, 4, 2, 1),
                        nn.ReLU(True)]

        decoder += [nn.ConvTranspose2d(dim, o_dim, 4, 2, 1),
                    nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z, t):
        o = torch.cat((z, t), 1)
        return self.decoder(o.reshape(o.shape[0], -1, 1, 1))
