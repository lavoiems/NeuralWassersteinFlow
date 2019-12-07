import os
import math
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pylab as plt


def spiral(scale, batch_size, translationx=0, translationy=0):
    while True:
        xy = []
        for _ in range(batch_size):
            t = random.uniform(0, 2) * math.pi
            x = translationx + (scale * t) * math.cos(t)
            y = translationy + (scale * t) * math.sin(t)
            xy.append((x, y))
        yield(torch.FloatTensor(xy))


def spiralplane(scale, batch_size, translationx=0, translationy=0):
    while True:
        xyz = []
        for _ in range(batch_size):
            t = random.uniform(0, 2) * math.pi
            x = translationx + (scale * t) * math.cos(t)
            y = translationy + (scale * t) * math.sin(t)
            z = random.random()
            xyz.append((x, y, z))
        yield(torch.FloatTensor(xyz))


def gaussians(batch_size):
    while True:
        points = np.random.randn(batch_size, 2) * 0.05
        points[:, 0] += 2 * np.random.randint(-2, high=3, size=batch_size)
        points[:, 1] += 2 * np.random.randint(-2, high=3, size=batch_size)
        yield(torch.from_numpy(points).type(torch.FloatTensor))


def gaussian1(batch_size, loc=0, scale=1):
    while True:
        points = np.stack([np.random.normal((loc,), (scale,)) for _ in range(batch_size)])
        yield(torch.from_numpy(points).type(torch.FloatTensor))


def gaussian2(batch_size, loc=0, scale=1):
    while True:
        points = np.stack([np.random.normal((loc, loc), (scale, scale)) for _ in range(batch_size)])
        yield(torch.from_numpy(points).type(torch.FloatTensor))


def mixture(batch_size):
    while True:
        points = []
        for _ in range(batch_size):
            if np.random.randint(0, 2):
                points.append(np.random.normal((10,), (2,)))
            else:
                points.append(np.random.normal((15,), (1,)))
        points = np.array(points)
        yield(torch.from_numpy(points).type(torch.FloatTensor))


def mixture2(batch_size):
    mean = np.array([[5, 5], [-5, -5]])
    var = np.array([[0.5, 0.5], [0.5, 0.5]])
    while True:
        points = np.stack([np.random.normal(mean[i%2], var[i%2]) for i in range(batch_size)])
        yield(torch.from_numpy(points).type(torch.FloatTensor))


def uniform(batch_size):
    while True:
        yield(torch.rand((batch_size, 2)))


def half_uniform(batch_size):
    while True:
        prior = torch.cat([torch.FloatTensor(batch_size//2, 2).uniform_(0, 0.5),
                           torch.FloatTensor(batch_size//2, 2).uniform_(0.5, 1)])
        yield(prior)


def half_uniform_img(batch_size):
    while True:
        prior = torch.cat([torch.FloatTensor(batch_size//2, 2).uniform_(0, 0.5),
                           torch.FloatTensor(batch_size//2, 2).uniform_(0.5, 1)])

        prior *= 20
        prior = prior.type(torch.ByteTensor)
        prior.clamp_(0, 19)
        img = torch.zeros(batch_size, 20, 20)
        for i, p in zip(img, prior):
            i[p[0].item(), p[1].item()] += 1
        yield(img.view(batch_size, -1))


def color_block(batch_size):
    while True:
        z = torch.rand(batch_size)
        prior = torch.zeros(batch_size, 2, 2, 3)
        prior[range(0, batch_size//2), 0, 0, 0] = z[:batch_size//2]
        prior[range(batch_size//2, batch_size), 1, 0, 1] = z[batch_size//2:]
        yield(prior.view(batch_size, -1))


def png(batch_size, filename):
    path = os.path.join(os.getcwd(), filename)
    image = Image.open(path).convert('RGB')
    image = np.array(image)[:,:,0]
    image = image < 128
    image = 255 * image
    imgsize = len(image)
    image = image.reshape(-1)
    px = image / image.sum()
    positions = np.arange(imgsize**2)
    positions = imgsize**2 - positions
    while True:
        pos = np.random.choice(positions, size=batch_size, p=px)
        x = torch.from_numpy(pos % imgsize)
        y = torch.from_numpy(pos // imgsize)
        data = torch.stack((x, y)).type(torch.FloatTensor)/imgsize
        data = data.transpose(0, 1)
        yield(data)


def pointcloud(batch_size, filename, idx):
    pc = np.load(filename)['pointcloud'][idx]
    npoints = len(pc)
    while True:
        idx = np.random.randint(0, npoints, size=batch_size)
        data = torch.from_numpy(pc[idx]).float()
        yield(data)


