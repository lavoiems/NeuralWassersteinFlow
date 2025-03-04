import time
import torch
from torch import optim
import matplotlib.pyplot as plt

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def disc_loss_generation(data, target, z_dim, eps, lp, critic1, critic2, generator, device):
    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z).detach()
    u = critic1(target)
    v = critic2(gen)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    target_ = target.view(data.shape[0], -1).unsqueeze(0)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    p = (u_ + v_ - (torch.abs(gen_ - target_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(2*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, target, z_dim, eps, lp, critic1, critic2, generator, device):
    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z)
    u = critic1(target)
    v = critic2(gen)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    target_ = target.view(data.shape[0], -1).unsqueeze(0)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    H = torch.clamp(u_ + v_ - (torch.abs(gen_ - target_)**lp).sum(2), 0)
    H = 1/eps*H
    loss = (torch.abs(target_ - gen_)**lp).sum(2)*H.detach()
    return 2*loss.mean()**2


def define_models(shape1, **parameters):
    criticx1 = model.Critic(shape1, **parameters)
    criticx2 = model.Critic(shape1, **parameters)
    criticy1 = model.Critic(shape1, **parameters)
    criticy2 = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'criticx1': criticx1,
        'criticx2': criticx2,
        'criticy1': criticy1,
        'criticy2': criticy2,
    }


@torch.no_grad()
def evaluate(visualiser, data, target1, target2, generator, id):
    fig = plt.figure()
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.scatter(*target1.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target1', f'{id}0')
    plt.clf()
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)

    plt.scatter(*target2.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target2', f'{id}0')
    plt.clf()
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    X = generator(data)
    plt.scatter(*X.cpu().numpy().transpose())
    visualiser.matplotlib(fig, f'data', f'{id}0')
    plt.clf()
    plt.close(fig)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    criticx1 = models['criticx1'].to(args.device)
    criticx2 = models['criticx2'].to(args.device)
    criticy1 = models['criticy1'].to(args.device)
    criticy2 = models['criticy2'].to(args.device)
    print(generator)
    print(criticx1)
    print(criticx2)
    print(criticy1)
    print(criticy2)

    optim_criticx1 = optim.Adam(criticx1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticx2 = optim.Adam(criticx2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy1 = optim.Adam(criticy1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy2 = optim.Adam(criticy2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2 = iter(test_loader1), iter(test_loader2)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        criticx1.train()
        criticx2.train()
        criticy1.train()
        criticy2.train()
        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx.to(args.device)
            optim_criticx1.zero_grad()
            optim_criticx2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, data, args.z_dim, args.eps, args.lp, criticx1, criticx2, generator, args.device)
            (r_loss + g_loss + p).backward(mone)
            optim_criticx1.step()
            optim_criticx2.step()

            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy.to(args.device)
            optim_criticy1.zero_grad()
            optim_criticy2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, datay, args.z_dim, args.eps, args.lp, criticy1, criticy2, generator, args.device)
            (r_loss + g_loss + p).backward(mone)
            optim_criticy1.step()
            optim_criticy2.step()

        optim_generator.zero_grad()
        t_lossx = transfer_loss(data, data, args.z_dim, args.eps, args.lp, criticx1, criticx2, generator, args.device)
        t_lossy = transfer_loss(data, datay, args.z_dim, args.eps, args.lp, criticy1, criticy2, generator, args.device)
        (0.5*t_lossx + 0.5*t_lossy).backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy.to(args.device)
            data = torch.randn(args.test_batch_size, args.z_dim, device=args.device)
            evaluate(args.visualiser, data, datax, datay, generator, 'x')
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss y')
            args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss x')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
