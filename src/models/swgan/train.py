import time
import torch
from torch import optim

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def disc_loss_generation(data, gen, eps, lp, critic1, critic2):
    u = critic1(data)
    v = critic2(gen)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    p = (u_ + v_ - (torch.abs(gen_ - data_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(2*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, eps, lp, nz, critic1, critic2, generator, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(z)
    u = critic1(data)
    v = critic2(gen)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    H = torch.clamp(u_ + v_ - (torch.abs(gen_ - data_)**lp).sum(2), 0)
    H = H/eps
    loss = (torch.abs(data_ - gen_)**lp).sum(2)*H.detach()
    return loss.mean()


def define_models(shape1, **parameters):
    critic1 = model.Critic(shape1[0], shape1[1], **parameters)
    critic2 = model.Critic(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'critic1': critic1,
        'critic2': critic2,
    }


@torch.no_grad()
def evaluate(visualiser, noise, data, transfer, id):
    transfered = transfer(noise).to('cpu').detach().numpy()
    visualiser.image(transfered, title=f'GAN generated', step=id)
    visualiser.image(data.cpu().numpy(), title=f'Target', step=id)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic1 = models['critic1'].to(args.device)
    critic2 = models['critic2'].to(args.device)
    print(generator)
    print(critic1)
    print(critic2)

    optim_critic1 = optim.Adam(critic1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic2 = optim.Adam(critic2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic1.train()
        critic2.train()
        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            if data.shape[0] != args.train_batch_size:
                batchx, iter1 = sample(iter1, train_loader1)
                data = batchx[0].to(args.device)

            optim_critic1.zero_grad()
            optim_critic2.zero_grad()
            z = torch.randn(data.shape[0], args.z_dim, device=args.device)
            gen1 = generator(z).detach()
            r_loss, g_loss, p = disc_loss_generation(data, gen1, args.eps, args.lp, critic1, critic2)
            (r_loss + g_loss + p).backward(mone)
            optim_critic1.step()
            optim_critic2.step()

        optim_generator.zero_grad()
        t_loss = transfer_loss(data, args.eps, args.lp, args.z_dim, critic1, critic2, generator, args.device)
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            noise = torch.randn(args.test_batch_size, args.z_dim, device=args.device)
            evaluate(args.visualiser, noise, data[:args.test_batch_size], generator, i)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss.detach().cpu().numpy(), title=f'Generator loss')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
