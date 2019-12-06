import time
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def compute_fc(x, y, lp, critic):
    x_ = x.view(x.shape[0], -1).unsqueeze(0)
    y_ = y.view(y.shape[0], -1).unsqueeze(1)

    cost = (x_ - y_).abs().pow(lp)
    cost = 1/lp*cost.sum(2)
    psi = (cost - critic(x).unsqueeze(0))
    return psi.min(1)[0]


def disc_loss_generation(data, target, eps, lp, critic):
    u = critic(data)
    v = compute_fc(data, target, lp, critic)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(1)
    target_ = target.view(data.shape[0], -1).unsqueeze(0)
    p = (u_ + v_ - (torch.abs(target_ - data_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(4*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, target, eps, lp, critic, generator, device):
    gen = generator(data)
    u = critic(data)
    v = compute_fc(data, target, lp, critic)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(1)
    target_ = target.view(target.shape[0], -1).unsqueeze(0)
    H = torch.clamp(u_ + v_ - (torch.abs(target_ - data_)**lp).sum(2), 0)
    H = 1/eps*H
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(0)
    loss = (torch.abs(target_ - gen_)**lp).sum(2)*H.detach()
    return loss.mean()


def define_models(shape1, **parameters):
    critic1 = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'critic1': critic1,
    }


@torch.no_grad()
def evaluate(visualiser, data, target, generator, id, device):
    fig = plt.figure()
    jet = plt.get_cmap('jet')
    alphas = data.sum(1)
    cNorm = colors.Normalize(vmin=alphas.min(), vmax=alphas.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color_val = scalarMap.to_rgba(alphas.cpu())

    plt.scatter(*data.cpu().numpy().transpose(), c=color_val)
    visualiser.matplotlib(fig, 'data', f'{id}0')
    plt.clf()

    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target', f'{id}0')
    plt.clf()

    transfered = generator(data).to('cpu').detach().numpy().transpose()
    plt.scatter(*transfered, c=color_val)
    visualiser.matplotlib(fig, 'Transfered', f'{id}0')
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
    print(generator)
    print(critic1)

    optim_critic1 = optim.Adam(critic1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1 = iter(test_loader1)
    titer2 = iter(test_loader2)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic1.train()
        batchx, iter1 = sample(iter1, train_loader1)
        data = batchx.to(args.device)
        batchy, iter2 = sample(iter2, train_loader2)
        target = batchy.to(args.device)
        for _ in range(args.d_updates):
            optim_critic1.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, target, args.eps, args.lp, critic1)
            (r_loss + g_loss + p).backward(mone)
            optim_critic1.step()

        optim_generator.zero_grad()
        t_loss = transfer_loss(data, target, args.eps, args.lp, critic1, generator, args.device)
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            data = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            target = batchy.to(args.device)
            evaluate(args.visualiser, data, target, generator, i, args.device)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss.detach().cpu().numpy(), title=f'Generator loss')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
