from torch import nn
import torch


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim, h_dim*2, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*2, h_dim*4, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*4, 4*h_dim, 4, 1, 0)]

        out = [nn.Linear(4*h_dim+3, h_dim),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Linear(h_dim, h_dim),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Linear(h_dim, h_dim),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Linear(h_dim, 1)]

        self.x = nn.Sequential(*x)
        self.out = nn.Sequential(*out)

    def forward(self, x, t):
        o = self.x(x).squeeze()
        o = torch.cat((o, t), 1)
        return self.out(o)


class Generator(nn.Module):
    def __init__(self, o_dim, h_dim, z_dim, **kwargs):
        super(Generator, self).__init__()

        x = [nn.ConvTranspose2d(z_dim+3, h_dim, 4, 1, 0),
             nn.ReLU(inplace=True),
             nn.ConvTranspose2d(h_dim, 2*h_dim, 4, 2, 1),
             nn.ReLU(inplace=True),
             nn.ConvTranspose2d(2*h_dim, 4*h_dim, 4, 2, 1),
             nn.ReLU(inplace=True),
             nn.ConvTranspose2d(4*h_dim, o_dim, 4, 2, 1),
             nn.Sigmoid()]
        self.x = nn.Sequential(*x)

    def forward(self, z, t):
        o = torch.cat((z, t), 1)
        return self.x(o)
