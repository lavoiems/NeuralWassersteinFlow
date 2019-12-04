import time
import torch
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def critic_loss(data, z_dim, critic, generator, device):
    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z).detach()
    f = critic(data).mean()

    fc = compute_fc(data, gen, 2, critic).mean()
    return fc + f


def transfer_loss(data, z_dim, critic, generator, device):
    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z)

    fc = compute_fc(data, gen, 2, critic).mean()
    return fc


def define_models(shape1, **parameters):
    critic = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'critic': critic,
    }


@torch.no_grad()
def evaluate(visualiser, nz, target, generator, id, device):
    fig = plt.figure()
    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target', f'{id}0')
    plt.clf()

    noise = torch.randn(target.shape[0], nz, device=device)
    transfered = generator(noise).to('cpu').detach().numpy().transpose()
    plt.scatter(*transfered)
    visualiser.matplotlib(fig, 'Transfered', f'{id}0')
    plt.clf()
    plt.close(fig)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic = models['critic'].to(args.device)
    print(generator)
    print(critic)

    optim_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1 = iter(test_loader1)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic.train()

        if i < 100:
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx.to(args.device)
            optim_generator.zero_grad()
            z = torch.randn(data.shape[0], args.z_dim, device=args.device)
            gen = generator(z)
            reg = F.mse_loss(gen, data)
            (10*reg).backward()
            optim_generator.step()

        else:
            for _ in range(args.d_updates):
                batchx, iter1 = sample(iter1, train_loader1)
                data = batchx.to(args.device)

                optim_critic.zero_grad()
                r_loss = critic_loss(data, args.z_dim, critic, generator, args.device)
                r_loss.backward(mone)
                optim_critic.step()

            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx.to(args.device)
            optim_generator.zero_grad()
            t_loss = transfer_loss(data, args.z_dim, critic, generator, args.device)
            t_loss.backward()
            optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            data = batchx.to(args.device)
            evaluate(args.visualiser, args.z_dim, data, generator, i, args.device)
            d_loss = (r_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss.detach().cpu().numpy(), title=f'Generator loss')
            #args.visualiser.plot(step=i, data=e_loss.detach().cpu().numpy(), title=f'Encoder loss')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
