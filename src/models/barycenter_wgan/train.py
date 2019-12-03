import time
import torch
from torch import optim
import matplotlib.pyplot as plt

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def gp_loss(x, y, d, device):
    batch_size = x.size()[0]
    gp_alpha = torch.rand(batch_size, 1, device=device)

    interp = gp_alpha * x.data + (1 - gp_alpha) * y.data
    interp.requires_grad = True
    d_interp = d(interp)
    grad_interp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                      grad_outputs=torch.ones(d_interp.size(), device=device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_interp = grad_interp.view(grad_interp.size(0), -1)
    diff = grad_interp.norm(2, dim=1) - 1
    diff = torch.clamp(diff, 0)
    return torch.mean(diff**2)


def disc_loss_generation(data, target, critic, generator, device):
    gen = generator(data).detach()
    u = critic(target).mean()
    v = critic(gen).mean()
    gp = gp_loss(gen, target, critic, device)
    return u, v, gp


def transfer_loss(data, target, critic, generator):
    gen = generator(data)
    u = critic(target).mean()
    v = critic(gen).mean()
    loss = (u - v).pow(2)
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
def evaluate(visualiser, data, data1, target, generator, id):
    fig = plt.figure()
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.scatter(*data1.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target1', f'{id}0')
    plt.clf()
    plt.xlim(-8,8)
    plt.ylim(-8,8)

    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target2', f'{id}0')
    plt.clf()
    plt.xlim(-8,8)
    plt.ylim(-8,8)
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
        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx.to(args.device)
            optim_criticx.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, data, criticx, generator, args.device)
            r_loss.backward()
            g_loss.backward(mone)
            p.backward()
            optim_criticx.step()

            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy.to(args.device)
            optim_criticy.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, datay, criticy, generator, args.device)
            r_loss.backward()
            g_loss.backward(mone)
            p.backward()
            optim_criticy.step()

        optim_generator.zero_grad()
        t_lossx = transfer_loss(data, data, criticx, generator)
        t_lossy = transfer_loss(data, datay, criticy, generator)
        (0.5*t_lossx + 0.5*t_lossy).backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy.to(args.device)
            evaluate(args.visualiser, datax, datax, datay, generator, 'x')
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss y')
            args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss x')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
