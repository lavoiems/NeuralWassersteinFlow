import time
import torch
from torch import optim

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def disc_loss_generation(data, nz, alpha, eps, lp, critic1, critic2, generator, device):
    t = torch.distributions.beta.Beta(alpha, alpha).sample_n(1).to(device)
    t = torch.stack([t]*data.shape[0])
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(z, t).detach()
    u = critic1(data, t)
    v = critic2(gen, t)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(1)
    p = (u_ + v_ - (torch.abs(gen_ - data_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(2*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, eps, lp, nz, t, critic1, critic2, generator, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(z, t)
    u = critic1(data, t)
    v = critic2(gen, t)
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
    critic3 = model.Critic(shape1[0], shape1[1], **parameters)
    critic4 = model.Critic(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'critic1': critic1,
        'critic2': critic2,
        'critic3': critic3,
        'critic4': critic4,
    }


@torch.no_grad()
def evaluate(visualiser, noise, data, data2, transfer, id, device):
    visualiser.image(data.cpu().numpy(), title=f'Target', step=id)
    visualiser.image(data2.cpu().numpy(), title=f'Target 2', step=id)

    card = 11
    for i in range(card):
        t = torch.FloatTensor([i/(card-1)]).repeat(data.shape[0], 1).to(device)
        X = transfer(noise, t).cpu().detach().numpy()
        visualiser.image(X, title=f'GAN generated {t[0]}', step=id)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic1 = models['critic1'].to(args.device)
    critic2 = models['critic2'].to(args.device)
    critic3 = models['critic3'].to(args.device)
    critic4 = models['critic4'].to(args.device)
    print(generator)
    print(critic1)
    print(critic2)

    optim_critic1 = optim.Adam(critic1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic2 = optim.Adam(critic2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic3 = optim.Adam(critic3.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic4 = optim.Adam(critic4.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic1.train()
        critic2.train()
        critic3.train()
        critic4.train()
        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            if data.shape[0] != args.train_batch_size:
                batchx, iter1 = sample(iter1, train_loader1)
                data = batchx[0].to(args.device)
            data[:,0] = data[:,0]*0
            data[:,1] = data[:,1]*0

            optim_critic1.zero_grad()
            optim_critic2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, args.z_dim, args.alpha, args.eps, args.lp, critic1, critic2, generator, args.device)
            (r_loss + g_loss + p).backward(mone)
            optim_critic1.step()
            optim_critic2.step()

            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy[0].to(args.device)
            if datay.shape[0] != args.train_batch_size:
                batchy, iter2 = sample(iter2, train_loader2)
                datay = batchy[0].to(args.device)
            datay[:,1] = datay[:,1]*0
            datay[:,2] = datay[:,2]*0

            optim_critic3.zero_grad()
            optim_critic4.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(datay, args.z_dim, args.alpha, args.eps, args.lp, critic3, critic4, generator, args.device)
            (r_loss + g_loss + p).backward(mone)
            optim_critic3.step()
            optim_critic4.step()

        optim_generator.zero_grad()
        t_ = torch.distributions.beta.Beta(args.alpha, args.alpha).sample_n(1).to(args.device)
        t = torch.stack([t_]*data.shape[0])
        t_loss1 = transfer_loss(data, args.eps, args.lp, args.z_dim, t, critic1, critic2, generator, args.device)
        t_loss2 = transfer_loss(datay, args.eps, args.lp, args.z_dim, t, critic3, critic4, generator, args.device)
        ((1-t_)*t_loss1 + t_*t_loss2).backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            noise = torch.randn(args.test_batch_size, args.z_dim, device=args.device)
            evaluate(args.visualiser, noise, data[:args.test_batch_size], datay[:args.test_batch_size], generator, i, args.device)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss1.detach().cpu().numpy(), title=f'Generator loss 1')
            args.visualiser.plot(step=i, data=t_loss2.detach().cpu().numpy(), title=f'Generator loss 2')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
