import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim, h_dim*2, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*2, h_dim*4, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*4, h_dim*4, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*4, 1, 4, 1, 0)]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        encoder = [nn.Conv2d(o_dim, h_dim, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(h_dim, h_dim*2, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(h_dim*2, h_dim*4, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(h_dim*4, h_dim*4, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(h_dim*4, h_dim*4, 4, 1, 0),
                   nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.ConvTranspose2d(h_dim*4+1, h_dim*4, 4, 1, 0),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(h_dim*4, h_dim*4, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(h_dim*4, h_dim*2, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(h_dim*2, h_dim, 4, 2, 1),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(h_dim, o_dim, 4, 2, 1),
                   nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z, t):
        o = self.encoder(z)
        o = torch.cat((o, t), 1)
        return self.decoder(o)
