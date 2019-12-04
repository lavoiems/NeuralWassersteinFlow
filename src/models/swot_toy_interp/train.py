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
    p = (u_ + v_ - (torch.abs(data_ - target_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(2*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, target, nt, t, eps, lp, critic, generator):
    gen = generator(data, t)
    u = critic(data)
    v = compute_fc(data, target, lp, critic)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(1)
    target_ = target.view(target.shape[0], -1).unsqueeze(0)
    H = torch.clamp(u_ + v_ - (torch.abs(data_ - target_)**lp).sum(2), 0)
    H = 1/eps*H
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    loss = (torch.abs(target_ - gen_)**lp).sum(2)*H.detach()
    loss = loss.view(nt, -1).mean(1)
    return loss


def define_models(shape1, **parameters):
    criticx = model.Critic(shape1, **parameters)
    criticy = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'criticx': criticx,
        'criticy': criticy,
    }


@torch.no_grad()
def evaluate(visualiser, data, data1, target, generator, id, device):
    fig = plt.figure()
    jet = plt.get_cmap('jet')
    alphas = data1.sum(1)
    cNorm = colors.Normalize(vmin=alphas.min(), vmax=alphas.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color_val = scalarMap.to_rgba(alphas.cpu())
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.scatter(*data1.cpu().numpy().transpose(), c=color_val)
    visualiser.matplotlib(fig, 'target1', f'{id}0')
    plt.clf()
    plt.xlim(-8,8)
    plt.ylim(-8,8)

    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target2', f'{id}0')
    plt.clf()
    card = 11
    for i in range(card):
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        t = torch.FloatTensor([i/(card-1)]).repeat(data.shape[0], 1).to(device)
        X = generator(data, t)
        plt.scatter(*X.cpu().numpy().transpose(), c=color_val)
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
    criticx = models['criticx'].to(args.device)
    criticy = models['criticy'].to(args.device)
    print(generator)
    print(criticx)
    print(criticy)

    optim_criticx = optim.Adam(criticx.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy = optim.Adam(criticy.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2 = iter(test_loader1), iter(test_loader2)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        criticx.train()
        criticy.train()

        batchx, iter1 = sample(iter1, train_loader1)
        data = batchx.to(args.device)
        batchy, iter2 = sample(iter2, train_loader2)
        datay = batchy.to(args.device)
        for _ in range(args.d_updates):
            optim_criticx.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, data, args.eps, args.lp, criticx)
            (r_loss + g_loss + p).backward(mone)
            optim_criticx.step()

            optim_criticy.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, datay, args.eps, args.lp, criticy)
            (r_loss + g_loss + p).backward(mone)
            optim_criticy.step()

        optim_generator.zero_grad()
        t_ = torch.distributions.beta.Beta(args.alpha, args.alpha).sample_n(args.nt).to(args.device)
        t = torch.stack([t_] * data.shape[0]).transpose(0, 1).reshape(-1, 1)
        tdata = torch.cat([data]*args.nt)
        tdatay = torch.cat([datay]*args.nt)
        t_lossx = transfer_loss(tdata, tdata, args.nt, t, args.eps, args.lp, criticx, generator)**args.p_exp
        t_lossy = transfer_loss(tdata, tdatay, args.nt, t, args.eps, args.lp, criticy, generator)**args.p_exp
        t_loss = ((1-t_)*t_lossx + t_*t_lossy).sum()
        t_loss.backward()
        optim_generator.step()
        t_loss = t_loss.detach().cpu().numpy()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy.to(args.device)
            evaluate(args.visualiser, datax, datax, datay, generator, 'x', args.device)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss, title=f'generator loss')
            with torch.no_grad():
                t_ = torch.arange(0, 1.1, 0.1, device=args.device)
                t = torch.stack([t_]*datax.shape[0]).transpose(0, 1).reshape(-1, 1)
                tdata = torch.cat([datax]*11)
                tdatay = torch.cat([datay]*11)
                t_lossx = transfer_loss(tdata, tdata, 11, t, args.eps, args.lp, criticx, generator)
                t_lossy = transfer_loss(tdata, tdatay, 11, t, args.eps, args.lp, criticy, generator)
                args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss x')
                args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss y')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
