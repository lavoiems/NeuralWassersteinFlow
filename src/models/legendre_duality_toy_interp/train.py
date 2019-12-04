import time
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import torch.nn.functional as F
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def compute_fc(x, y, lp, t, critic):
    x_ = x.view(x.shape[0], -1).unsqueeze(0)
    y_ = y.view(y.shape[0], -1).unsqueeze(1)

    cost = (x_ - y_).abs().pow(lp)
    cost = 1/lp*cost.sum(2)
    psi = (cost - critic(x, t).unsqueeze(0))
    return psi.min(1)[0]


def critic_loss(data1, data2, alpha, critic, generator, device):
    #t_ = torch.distributions.beta.Beta(alpha, alpha).sample_n(1).to(device)
    t_ = torch.FloatTensor([1]).to(device)
    t = torch.stack([t_]*data1.shape[0])
    f = critic(data2, t).mean()
    gen = generator(data1, t).detach()
    fc = compute_fc(data2, gen, 2, t, critic).mean()
    return fc + f


def transfer_loss(z, data, t, critic, generator, device):
    f = critic(data, t).mean()
    gen = generator(z, t)
    fc = compute_fc(data, gen, 2, t, critic).mean()
    return fc + f


def define_models(shape1, **parameters):
    critic1 = model.Critic(shape1, **parameters)
    critic2 = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'critic1': critic1,
        'critic2': critic2,
    }


def visualize_points(visualiser, fig, points, idx, bound=0.5, c=None):
    #fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0:
        ax.scatter(*points, c=c)

    ax.set_title('Points')
    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)
    ax.view_init(elev=30, azim=45)
    visualiser.matplotlib(fig, f'target 1 {idx}', f'{id}0')
    plt.clf()


@torch.no_grad()
def evaluate(visualiser, target1, target2, z_dim, generator, id, device):
    fig = plt.figure()

    jet = plt.get_cmap('jet')
    alphas = target1.sum(1)
    cNorm = colors.Normalize(vmin=alphas.min(), vmax=alphas.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color_val = scalarMap.to_rgba(alphas.cpu())
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    #visualize_points(visualiser, fig, target1.cpu().numpy().transpose(), 'start', c=color_val, bound=0.5)
    #visualize_points(visualiser, fig, target2.cpu().numpy().transpose(), 'end', bound=0.5)

    plt.scatter(*target1.cpu().numpy().transpose(), c=color_val)
    visualiser.matplotlib(fig, 'target 1', f'{id}0')
    plt.clf()

    fig = plt.figure()
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.scatter(*target2.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target 2', f'{id}0')
    plt.clf()

    card = 11
    z = target1
    #z = torch.randn(target1.shape[0], z_dim, device=device)
    for i in range(card):
        t = torch.FloatTensor([i/(card-1)]).repeat(target1.shape[0], 1).to(device)
        #t = torch.FloatTensor([1]).repeat(target1.shape[0], 1).to(device)
        X = generator(z, t).cpu().numpy().transpose()
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        #visualize_points(visualiser, fig, X, t, c=color_val, bound=0.5)
        plt.scatter(*X, c=color_val)
        visualiser.matplotlib(fig, f'data{i}', f'{id}0')
        plt.clf()
    plt.close(fig)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic1 = models['critic1'].to(args.device)
    critic2 = models['critic2'].to(args.device)
    print(generator)
    print(critic1)

    optim_critic1 = optim.Adam(critic1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic2 = optim.Adam(critic2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2 = iter(test_loader1), iter(test_loader2)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic1.train()
        critic2.train()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx.to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy.to(args.device)

            optim_critic1.zero_grad()
            r_loss1 = critic_loss(data1, data1, args.alpha, critic1, generator, args.device)
            r_loss1.backward(mone)
            optim_critic1.step()

            optim_critic2.zero_grad()
            r_loss2 = critic_loss(data1, data2, args.alpha, critic2, generator, args.device)
            r_loss2.backward(mone)
            optim_critic2.step()

        optim_generator.zero_grad()
        for _ in range(10):
            t_ = torch.distributions.beta.Beta(args.alpha, args.alpha).sample_n(1).to(args.device)
            t = torch.stack([t_]*data1.shape[0])
            t_loss1 = transfer_loss(data1, data1, t, critic1, generator, args.device)**2
            t_loss2 = transfer_loss(data1, data2, t, critic2, generator, args.device)**2
            ((1-t_)*t_loss1 + t_*t_loss2).backward()

        #t_ = torch.FloatTensor([0]).to(args.device)
        #t = torch.stack([t_]*data1.shape[0])
        #gen = generator(data1, t)
        #reg = F.mse_loss(gen, data1)
        #(10*reg).backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            data1 = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            data2 = batchy.to(args.device)
            evaluate(args.visualiser, data1, data2, args.z_dim, generator, i, args.device)
            args.visualiser.plot(step=i, data=r_loss1.detach().cpu().numpy(), title=f'Critic loss 1')
            args.visualiser.plot(step=i, data=r_loss2.detach().cpu().numpy(), title=f'Critic loss 2')
            args.visualiser.plot(step=i, data=t_loss1.detach().cpu().numpy(), title=f'Generator loss 1')
            args.visualiser.plot(step=i, data=t_loss2.detach().cpu().numpy(), title=f'Generator loss 2')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
