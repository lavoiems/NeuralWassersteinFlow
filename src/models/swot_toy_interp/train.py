import time
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def disc_loss_generation(data, target, eps, lp, critic1, critic2):
    u = critic1(data)
    v = critic2(target)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    target_ = target.view(target.shape[0], -1).unsqueeze(1)
    p = (u_ + v_ - (torch.abs(data_ - target_)**lp).sum(2))
    p.clamp_(0)
    p = -(1/(2*eps))*p**2
    return u.mean(), v.mean(), p.mean()


def transfer_loss(data, target, nt, t, eps, lp, critic1, critic2, generator):
    gen = generator(data, t)
    u = critic1(data)
    v = critic2(target)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    target_ = target.view(target.shape[0], -1).unsqueeze(1)
    H = torch.clamp(u_ + v_ - (torch.abs(data_ - target_)**lp).sum(2), 0)
    H = H/(2*eps)
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(0)
    loss = (torch.abs(target_ - gen_)**lp).sum(2)*H.detach()
    loss = loss.view(nt, -1).mean(1)
    return loss


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
def evaluate(visualiser, data, target, generator, id, device):
    fig = plt.figure()
    jet = plt.get_cmap('jet')
    alphas = data.sum(1)
    cNorm = colors.Normalize(vmin=alphas.min(), vmax=alphas.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color_val = scalarMap.to_rgba(alphas.cpu())
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(*data.cpu().numpy().transpose(), c=color_val)
    visualiser.matplotlib(fig, 'target1', f'{id}0')
    plt.clf()
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target2', f'{id}0')
    plt.clf()
    card = 11
    for i in range(card):
        plt.xlim(0,1)
        plt.ylim(0,1)
        t_ = torch.FloatTensor([1]).to(device)
        t = torch.stack([t_] * data.shape[0]).transpose(0, 1).reshape(-1, 1)
        #t = torch.FloatTensor([i/(card-1)]).repeat(data.shape[0], 1).to(device)
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
    #criticx1 = models['criticx1'].to(args.device)
    #criticx2 = models['criticx2'].to(args.device)
    criticy1 = models['criticy1'].to(args.device)
    criticy2 = models['criticy2'].to(args.device)
    print(generator)
    #print(criticx1)
    print(criticy1)

    #optim_criticx1 = optim.Adam(criticx1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    #optim_criticx2 = optim.Adam(criticx2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
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
        #criticx1.train()
        #criticx2.train()
        criticy1.train()
        criticy2.train()

        for _ in range(args.d_updates):
            #batchx, iter1 = sample(iter1, train_loader1)
            #data = batchx.to(args.device)
            batchx, iter1 = sample(iter1, train_loader1)
            input_data = batchx.to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy.to(args.device)

            #optim_criticx1.zero_grad()
            #optim_criticx2.zero_grad()
            #r_loss, g_loss, p = disc_loss_generation(input_data, data, args.eps, args.lp, criticx1, criticx2)
            #(r_loss + g_loss + p).backward(mone)
            #optim_criticx1.step()
            #optim_criticx2.step()

            optim_criticy1.zero_grad()
            optim_criticy2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(input_data, datay, args.eps, args.lp, criticy1, criticy2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticy1.step()
            optim_criticy2.step()

        optim_generator.zero_grad()
        #t_ = torch.randn(args.nt, device=args.device)
        t_ = torch.FloatTensor([1]*args.nt).to(args.device)
        t = torch.stack([t_]*input_data.shape[0])
        #t = torch.stack([t_] * input_data.shape[0]).transpose(0, 1).reshape(-1, 1)
        tinputdata = torch.cat([input_data]*args.nt)
        #tdata = torch.cat([data]*args.nt)
        tdatay = torch.cat([datay]*args.nt)
        #t_lossx = transfer_loss(tinputdata, tdata, args.nt, t, args.eps, args.lp, criticx1, criticx2, generator)**args.p_exp
        t_lossy = transfer_loss(tinputdata, tdatay, args.nt, t, args.eps, args.lp, criticy1, criticy2, generator)**args.p_exp
        t_lossy.backward()
        #t_loss = ((1-t_)*t_lossx + t_*t_lossy).sum()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy.to(args.device)
            evaluate(args.visualiser, datax, datay, generator, 'x', args.device)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss Y')
            args.visualiser.plot(step=i, data=t_lossy, title=f'Generator loss Y')
            #args.visualiser.plot(step=i, data=t_loss, title=f'generator loss')
            #with torch.no_grad():
                #t_ = torch.arange(0, 1.1, 0.1, device=args.device)
                #t = torch.stack([t_]*datax.shape[0]).transpose(0, 1).reshape(-1, 1)
                #tinputdata = torch.cat([input_data]*11)
                #tdata = torch.cat([datax]*11)
                #tdatay = torch.cat([datay]*11)
                #t_lossx = transfer_loss(tinputdata, tdata, 11, t, args.eps, args.lp, criticx1, criticx2, generator)
                #t_lossy = transfer_loss(tinputdata, tdatay, 11, t, args.eps, args.lp, criticy1, criticy2, generator)
                #args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss x')
                #args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss y')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
