import torch
from torch import nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, i_dim, h_dim, **kwargs):
        super(Critic, self).__init__()

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim, h_dim*2, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*2, h_dim*4, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim*4, 1, 4, 1, 0)]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.tcbn = TCBN(h_dim, out_dim)

    def forward(self, x, t):
        o = self.conv(x)
        o = self.tcbn(o, t[0])
        return F.relu(o)


class Generator(nn.Module):
    def __init__(self, o_dim, h_dim, **kwargs):
        super(Generator, self).__init__()

        x = [nn.Conv2d(o_dim, h_dim, 4, 2, 1),
             nn.ReLU(inplace=True),
             nn.Conv2d(h_dim, h_dim*2, 4, 2, 1),
             nn.ReLU(inplace=True),
             nn.Conv2d(h_dim*2, h_dim*4, 4, 2, 1),
             nn.ReLU(inplace=True),
             nn.Conv2d(h_dim*4, 4*h_dim, 4, 1, 0)]

        self.x = nn.Sequential(*x)

        out = [nn.Conv2d(4*h_dim+3, 4*h_dim, 1, 1, 0),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(4*h_dim, 4*h_dim, 4, 1, 0),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(4*h_dim, 2*h_dim, 4, 2, 1),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(2*h_dim, h_dim, 4, 2, 1),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(h_dim, o_dim, 4, 2, 1),
               nn.Sigmoid()]
        self.out = nn.Sequential(*out)

    def forward(self, z, t):
        o = self.x(z).squeeze()
        o = torch.cat((o, t), 1)
        o = o.view(o.shape[0], -1, 1, 1)
        return self.out(o)


class TCBN(nn.Module):
    def __init__(self, n_hidden, num_features, eps=1e-5):
        super(TCBN, self).__init__()
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(4, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, num_features),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(4, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, num_features),
        )
        self.mean = torch.FloatTensor([0])
        self.var = torch.FloatTensor([1])

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def forward(self, input, t):
        # Obtain delta values from MLP
        gamma = 1-self.fc_gamma(t)
        beta = self.fc_beta(t)

        # Standard batch normalization
        return F.batch_norm(input, None, None, gamma, beta, True, 0, self.eps)
