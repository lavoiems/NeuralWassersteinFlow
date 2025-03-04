import math
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             #nn.BatchNorm2d(h_dim),
             nn.LeakyReLU(0.2, inplace=True)]

        dim = h_dim
        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        for _ in range(n_layers):
            in_dim = dim
            dim *= 2
            x += [nn.Conv2d(in_dim, dim, 4, 2, 1),
                  #nn.BatchNorm2d(dim),
                  nn.LeakyReLU(0.2, inplace=True)]
        x += [nn.Conv2d(dim, 1, 4, 1, 0)]
        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Generator(nn.Module):
    def __init__(self, o_dim, kernel_dim, z_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        dim = h_dim * n_layers**2
        decoder = [nn.ConvTranspose2d(z_dim, dim, 4, 1, 0),
                   nn.BatchNorm2d(dim),
                   nn.ReLU(True)]

        for _ in range(n_layers):
            in_dim = dim
            dim //= 2
            decoder += [nn.ConvTranspose2d(in_dim, dim, 4, 2, 1),
                        nn.BatchNorm2d(dim),
                        nn.ReLU(inplace=True)]

        decoder += [nn.ConvTranspose2d(dim, o_dim, 4, 2, 1),
                    nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z):
        return self.decoder(z.reshape(z.shape[0], z.shape[1], 1, 1))


class Encoder(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, **kwargs):
        super(Encoder, self).__init__()
        assert kernel_dim % 16 == 0, "kernel_dim has to be a multiple of 16"

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.BatchNorm2d(h_dim),
             nn.ReLU(inplace=True)]

        dim = h_dim
        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        for _ in range(n_layers):
            in_dim = dim
            dim *= 2
            x += [nn.Conv2d(in_dim, dim, 4, 2, 1),
                  nn.BatchNorm2d(dim),
                  nn.ReLU(inplace=True)]
        x += [nn.Conv2d(dim, dim, 4, 1, 0),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(dim, dim, 4, 1, 0),
              nn.BatchNorm2d(dim),
              nn.ReLU(inplace=True)]

        for _ in range(n_layers):
            in_dim = dim
            dim //= 2
            x += [nn.ConvTranspose2d(in_dim, dim, 4, 2, 1),
                  nn.BatchNorm2d(dim),
                  nn.ReLU(inplace=True)]
        x += [nn.ConvTranspose2d(dim, i_dim, 4, 2, 1),
              nn.Sigmoid()]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()

