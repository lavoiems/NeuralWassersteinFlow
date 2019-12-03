import time
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def encoder_loss(data, alpha, encoder, critic, generator, device):
    t_ = torch.distributions.beta.Beta(alpha, alpha).sample_n(1).to(device)
    t = torch.stack([t_]*data.shape[0])
    enc = encoder(data)
    gen = generator(enc, t)
    return (critic(gen, t) + (gen.view(gen.shape[0], -1) - data.view(data.shape[0], -1)).pow(2).sum(1)).mean()


def critic_loss(data, z_dim, alpha, encoder, critic, generator, device):
    t_ = torch.distributions.beta.Beta(alpha, alpha).sample_n(1).to(device)
    t = torch.stack([t_]*data.shape[0])
    enc = encoder(data)
    gen = generator(enc, t).detach()
    fc = (critic(gen, t) + (gen.view(gen.shape[0], -1) - data.view(data.shape[0], -1)).pow(2).sum(1)).mean()

    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z, t).detach()
    f = critic(gen, t).mean()

    return fc - f


def transfer_loss(data, z_dim, t, encoder, critic, generator, device):
    enc = encoder(data).detach()
    gen = generator(enc, t)
    fc = (critic(gen, t) + (gen.view(gen.shape[0], -1) - data.view(data.shape[0], -1)).pow(2).sum(1)).mean()

    z = torch.randn(data.shape[0], z_dim, device=device)
    gen = generator(z, t)
    f = critic(gen, t).mean()

    return fc - f


def define_models(shape1, **parameters):
    critic1 = model.Critic(shape1[0], shape1[1], **parameters)
    critic2 = model.Critic(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    encoder1 = model.Encoder(shape1[0], shape1[1], **parameters)
    encoder2 = model.Encoder(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'critic1': critic1,
        'critic2': critic2,
        'encoder1': encoder1,
        'encoder2': encoder2,
    }


@torch.no_grad()
def evaluate(visualiser, data, data2, transfer, encoder, id, device):
    visualiser.image(data.cpu().numpy(), title=f'Target', step=id)
    visualiser.image(data2.cpu().numpy(), title=f'Target 2', step=id)

    card = 11
    for i in range(card):
        t = torch.FloatTensor([i/(card-1)]).repeat(data.shape[0], 1).to(device)
        enc = encoder(data)
        X = transfer(enc, t).cpu().detach().numpy()
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
    encoder1 = models['encoder1'].to(args.device)
    encoder2 = models['encoder2'].to(args.device)
    print(generator)
    print(critic1)
    print(encoder1)

    optim_critic1 = optim.Adam(critic1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_critic2 = optim.Adam(critic2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_encoder1 = optim.Adam(encoder1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_encoder2 = optim.Adam(encoder2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2 = iter(test_loader1), iter(test_loader2)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic1.train()
        critic2.train()
        encoder1.train()
        encoder2.train()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)
            if data1.shape[0] != args.train_batch_size:
                continue
            if data2.shape[0] != args.train_batch_size:
                continue

            optim_critic1.zero_grad()
            r_loss1 = critic_loss(data1, args.z_dim, args.alpha, encoder1, critic1, generator, args.device)
            r_loss1.backward(mone)
            optim_critic1.step()

            optim_critic2.zero_grad()
            r_loss2 = critic_loss(data2, args.z_dim, args.alpha, encoder2, critic2, generator, args.device)
            r_loss2.backward(mone)
            optim_critic2.step()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)
            if data1.shape[0] != args.train_batch_size:
                continue
            if data2.shape[0] != args.train_batch_size:
                continue

            optim_encoder1.zero_grad()
            e_loss1 = encoder_loss(data1, args.alpha, encoder1, critic1, generator, args.device)
            e_loss1.backward()
            optim_encoder1.step()

            optim_encoder2.zero_grad()
            e_loss2 = encoder_loss(data2, args.alpha, encoder2, critic2, generator, args.device)
            e_loss2.backward()
            optim_encoder2.step()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)
            if data1.shape[0] != args.train_batch_size:
                continue
            if data2.shape[0] != args.train_batch_size:
                continue
            optim_generator.zero_grad()
            t_ = torch.distributions.beta.Beta(args.alpha, args.alpha).sample_n(1).to(args.device)
            t = torch.stack([t_]*data1.shape[0])
            t_loss1 = transfer_loss(data1, args.z_dim, t, encoder1, critic1, generator, args.device)
            t_loss2 = transfer_loss(data2, args.z_dim, t, encoder2, critic2, generator, args.device)
            ((1-t_)*t_loss1 + t_*t_loss2).backward()
            optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            encoder1.eval()
            encoder2.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            data1 = batchx[0].to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            data2 = batchy[0].to(args.device)
            evaluate(args.visualiser, data1, data2, generator, encoder1, i, args.device)
            d_loss = (r_loss1).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss')
            args.visualiser.plot(step=i, data=t_loss1.detach().cpu().numpy(), title=f'Generator loss 1')
            args.visualiser.plot(step=i, data=t_loss2.detach().cpu().numpy(), title=f'Generator loss 2')
            args.visualiser.plot(step=i, data=e_loss1.detach().cpu().numpy(), title=f'Encoder loss 1')
            args.visualiser.plot(step=i, data=e_loss2.detach().cpu().numpy(), title=f'Encoder loss 2')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
