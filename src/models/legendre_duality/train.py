import time
import torch
from torch import optim
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import torch.nn.functional as F

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def c_transform(y, ey, lp, critic):
    cy = critic(ey)
    cost = (ey.view(ey.shape[0], -1) - y.view(y.shape[0], -1)).abs().pow(lp).sum(1)
    return (cy - cost).mean()


def encoder_loss(batch_size, lp, z_dim, encoder, generator, critic, device):
    z = torch.randn(batch_size, z_dim, device=device)
    y = generator(z).detach()
    ey = encoder(y)
    return c_transform(y, ey, lp, critic)


def critic_loss(x, lp, z_dim, encoder, critic, generator, device):
    f = critic(x).mean()
    z = torch.randn(x.shape[0], z_dim, device=device)
    y = generator(z).detach()
    ey = encoder(y).detach()
    return f - critic(e(y))


def transfer_loss(batch_size, lp, z_dim, encoder, critic, generator, device):
    z = torch.randn(batch_size, z_dim, device=device)
    y = generator(z)
    ey = encoder(y).detach()
    return -c_transform(y, ey, lp, critic)


def define_models(shape1, **parameters):
    critic = model.Critic(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    encoder = model.Encoder(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'critic': critic,
        'encoder': encoder,
    }


def evaluate(visualiser, nz, data, encoder, generator, critic, z_dim, id, device):
    z = torch.randn(data.shape[0], nz, device=device)
    z.requires_grad = True
    dec = generator(z)
    visualiser.image(dec.cpu().detach().numpy(), title=f'GAN generated', step=id)
    visualiser.image(data.cpu().numpy(), title=f'Target', step=id)

    enc = encoder(dec)
    visualiser.image(enc.cpu().detach().numpy(), title=f'GAN encoded', step=id)


@torch.no_grad()
def evaluate_clusters(visualiser, encoder, target, label, id):
    enc = encoder(target)
    pca = PCA(2)
    emb = pca.fit_transform(enc.reshape(enc.shape[0], -1).cpu().squeeze().numpy())
    fig = plt.figure()
    colors = [f'C{c}' for c in label.cpu().numpy()]
    plt.scatter(*emb.transpose(), c=colors)
    visualiser.matplotlib(fig, f'Embeddings {id}', None)
    plt.clf()
    plt.close(fig)


@torch.no_grad()
def evaluate_distance(visualiser, encoder, loader1, loader2, device):
    ds = torch.zeros(10, 10, device=device)
    totals = torch.zeros(10, 10, device=device)
    for b1, b2 in zip(loader1, loader2):
        d1, d2 = b1[0].to(device), b2[0].to(device)
        l1, l2 = b1[1].to(device), b2[1].to(device)
        z1, z2 = encoder(d1), encoder(d2)
        dist = F.pairwise_distance(z1, z2, 2)
        ds[l1, l2] += dist
        ds[l2, l1] += dist
        totals[l1, l2] += 1
        totals[l2, l1] += 1
    avgs = ds / totals

    fig, ax = plt.subplots()
    im = ax.imshow(avgs.cpu().numpy())
    for i, row in enumerate(avgs):
        for j, point in enumerate(row):
            text = ax.text(j, i, f'{point.cpu().item():.3f}', ha='center', va='center', color='w', size=6)
    visualiser.matplotlib(fig, 'distances', None)
    plt.clf()
    plt.close(fig)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic = models['critic'].to(args.device)
    encoder = models['encoder'].to(args.device)
    print(generator)
    print(critic)
    print(encoder)

    optim_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_encoder = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1 = iter(test_loader1)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic.train()
        encoder.train()

        for _ in range(10):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            optim_encoder.zero_grad()
            optim_generator.zero_grad()
            e_loss = encoder_loss(data.shape[0], args.lp, args.z_dim, encoder, generator, critic, args.device)
            e_loss.backward()
            optim_encoder.step()
            optim_generator.step()

        for _ in range(1):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            optim_critic.zero_grad()
            r_loss = critic_loss(data, args.lp, args.z_dim, encoder, critic, generator, args.device)
            r_loss.backward(mone)
            optim_critic.step()

        for _ in range(1):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            optim_generator.zero_grad()
            t_loss = transfer_loss(data.shape[0], args.lp, args.z_dim, encoder, critic, generator, args.device)
            t_loss.backward()
            optim_generator.step()

        if i % args.evaluate == 0:
            encoder.eval()
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            data = batchx[0].to(args.device)
            evaluate(args.visualiser, args.z_dim, data, encoder, generator, critic, args.z_dim, i, args.device)
            d_loss = (r_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=e_loss.detach().cpu().numpy(), title=f'Encoder loss')
            args.visualiser.plot(step=i, data=t_loss.detach().cpu().numpy(), title=f'Generator loss')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
