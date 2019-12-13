import time
import torch
from torch import optim
from torch.distributions.dirichlet import Dirichlet

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


def transfer_loss(data, target, t, eps, lp, critic1, critic2, generator):
    gen = generator(data, t)
    u = critic1(data)
    v = critic2(target)
    u_ = u.unsqueeze(0)
    v_ = v.unsqueeze(1)
    data_ = data.view(data.shape[0], -1).unsqueeze(0)
    target_ = target.view(target.shape[0], -1).unsqueeze(1)
    H = torch.clamp(u_ + v_ - (torch.abs(data_ - target_)**lp).sum(2), 0)
    H = H/eps
    gen_ = gen.view(gen.shape[0], -1).unsqueeze(0)
    loss = (torch.abs(target_ - gen_)**lp).sum(2)*H.detach()
    return loss.mean()


def define_models(shape1, **parameters):
    criticx1 = model.Critic(shape1[0], **parameters)
    criticx2 = model.Critic(shape1[0], **parameters)
    criticy1 = model.Critic(shape1[0], **parameters)
    criticy2 = model.Critic(shape1[0], **parameters)
    criticz1 = model.Critic(shape1[0], **parameters)
    criticz2 = model.Critic(shape1[0], **parameters)
    generator = model.Generator(shape1[0], **parameters)
    return {
        'generator': generator,
        'criticx1': criticx1,
        'criticx2': criticx2,
        'criticy1': criticy1,
        'criticy2': criticy2,
        'criticz1': criticz1,
        'criticz2': criticz2,
    }


@torch.no_grad()
def evaluate(visualiser, data, target, target2, target3, generator, id, device):
    visualiser.image(target.cpu().numpy(), title=f'Target', step=id)
    visualiser.image(data.cpu().numpy(), title=f'Source', step=id)
    visualiser.image(target2.cpu().numpy(), title=f'Target 2', step=id)
    visualiser.image(target3.cpu().numpy(), title=f'Target 3', step=id)
    data12 = torch.clone(data)
    data12[:,0] = data12[:,0]*0
    data02 = torch.clone(data)
    data02[:,1] = data02[:,1]*0
    data01 = torch.clone(data)
    data01[:,2] = data01[:2]*0
    visualiser.image(data12.cpu().numpy(), title=f'Data [0, 1, 1]', step=id)
    visualiser.image(data02.cpu().numpy(), title=f'Data [1, 0, 1]', step=id)
    visualiser.image(data01.cpu().numpy(), title=f'Data [1, 1, 0]', step=id)

    concentrations = [(1,0,0),
                      (0.8, 0, 0.2), (0.8, 0, 0.2),
                      (0.6, 0, 0.4), (0.5, 0.25, 0.25), (0.6, 0.4, 0),
                      (0.4, 0, 0.6), (0.25, 0.25, 0.6), (0.34, 0.33, 0.33), (0.4, 0.6, 0),
                      (0.2, 0, 0.8), (0.2, 0.2, 0.6), (0.25, 0.25, 0.5), (0.25, 0.5, 0.25), (0.2, 0.8, 0),
                      (0, 0, 1), (0, 0.2, 0.8), (0, 0.4, 0.6), (0, 0.6, 0.4), (0, 0.8, 0.2), (0, 1, 0)]
    for t_ in concentrations:
        t = torch.stack([torch.FloatTensor([t_])] * data.shape[0]).to(device)
        X = generator(data, t)
        visualiser.image(X.cpu().numpy(), title=f'Generated {t_}', step=id)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2
    train_loader3, test_loader3 = args.loaders3
    train_loader4, test_loader4 = args.loaders4

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    criticx1 = models['criticx1'].to(args.device)
    criticx2 = models['criticx2'].to(args.device)
    criticy1 = models['criticy1'].to(args.device)
    criticy2 = models['criticy2'].to(args.device)
    criticz1 = models['criticz1'].to(args.device)
    criticz2 = models['criticz2'].to(args.device)
    print(generator)
    print(criticx1)
    print(criticy1)

    optim_criticx1 = optim.Adam(criticx1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticx2 = optim.Adam(criticx2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy1 = optim.Adam(criticy1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy2 = optim.Adam(criticy2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticz1 = optim.Adam(criticz1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticz2 = optim.Adam(criticz2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2, iter3, iter4 = iter(train_loader1), iter(train_loader2), iter(train_loader3), iter(train_loader4)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2, titer3, titer4 = iter(test_loader1), iter(test_loader2), iter(test_loader3), iter(test_loader4)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()

    generator.train()
    criticx1.train()
    criticx2.train()
    criticy1.train()
    criticy2.train()
    criticz1.train()
    criticz2.train()
    for i in range(5000):
        batchx, iter1 = sample(iter1, train_loader1)
        data = batchx[0].to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        datay = batchy[0].to(args.device)
        datay[:,1] = datay[:,1]*0
        datay[:,2] = datay[:,2]*0

        batchz, iter3 = sample(iter3, train_loader3)
        dataz = batchz[0].to(args.device)
        dataz[:,0] = dataz[:,0]*0
        dataz[:,2] = dataz[:,2]*0

        batchw, iter4 = sample(iter4, train_loader4)
        dataw = batchw[0].to(args.device)
        dataw[:,0] = dataw[:,0]*0
        dataw[:,1] = dataw[:,1]*0

        optim_criticx1.zero_grad()
        optim_criticx2.zero_grad()
        r_loss, g_loss, p = disc_loss_generation(data, datay, args.eps, args.lp, criticx1, criticx2)
        (r_loss + g_loss + p).backward(mone)
        optim_criticx1.step()
        optim_criticx2.step()

        optim_criticy1.zero_grad()
        optim_criticy2.zero_grad()
        r_loss, g_loss, p = disc_loss_generation(data, dataz, args.eps, args.lp, criticy1, criticy2)
        (r_loss + g_loss + p).backward(mone)
        optim_criticy1.step()
        optim_criticy2.step()

        optim_criticz1.zero_grad()
        optim_criticz2.zero_grad()
        r_loss, g_loss, p = disc_loss_generation(data, dataw, args.eps, args.lp, criticz1, criticz2)
        (r_loss + g_loss + p).backward(mone)
        optim_criticz1.step()
        optim_criticz2.step()

        if i % 100 == 0:
            print(f'Critics-{i}')
            print('Iter: %s' % i, time.time() - t0)
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss Y')
            t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        criticx1.train()
        criticx2.train()
        criticy1.train()
        criticy2.train()
        criticz1.train()
        criticz2.train()

        for i in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)

            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy[0].to(args.device)
            datay[:,1] = datay[:,1]*0
            datay[:,2] = datay[:,2]*0

            batchz, iter3 = sample(iter3, train_loader3)
            dataz = batchz[0].to(args.device)
            dataz[:,0] = dataz[:,0]*0
            dataz[:,2] = dataz[:,2]*0

            batchw, iter4 = sample(iter4, train_loader4)
            dataw = batchw[0].to(args.device)
            dataw[:,0] = dataw[:,0]*0
            dataw[:,1] = dataw[:,1]*0

            optim_criticx1.zero_grad()
            optim_criticx2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, datay, args.eps, args.lp, criticx1, criticx2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticx1.step()
            optim_criticx2.step()

            optim_criticy1.zero_grad()
            optim_criticy2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, dataz, args.eps, args.lp, criticy1, criticy2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticy1.step()
            optim_criticy2.step()

            optim_criticz1.zero_grad()
            optim_criticz2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(data, dataw, args.eps, args.lp, criticz1, criticz2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticz1.step()
            optim_criticz2.step()

        batchx, iter1 = sample(iter1, train_loader1)
        data = batchx[0].to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        datay = batchy[0].to(args.device)
        datay[:,1] = datay[:,1]*0
        datay[:,2] = datay[:,2]*0

        batchz, iter3 = sample(iter3, train_loader3)
        dataz = batchz[0].to(args.device)
        dataz[:,0] = dataz[:,0]*0
        dataz[:,2] = dataz[:,2]*0

        batchw, iter4 = sample(iter4, train_loader4)
        dataw = batchw[0].to(args.device)
        dataw[:,0] = dataw[:,0]*0
        dataw[:,1] = dataw[:,1]*0

        optim_generator.zero_grad()
        t_ = Dirichlet(torch.FloatTensor([1.,1.,1.])).sample().to(args.device)
        t = torch.stack([t_]*data.shape[0])
        t_lossx = transfer_loss(data, data, t, args.eps, args.lp, criticx1, criticx2, generator)
        t_lossy = transfer_loss(data, datay, t, args.eps, args.lp, criticy1, criticy2, generator)
        t_lossz = transfer_loss(data, dataz, t, args.eps, args.lp, criticz1, criticz2, generator)
        t_loss = (t_[0]*t_lossx + t_[1]*t_lossy + t_[2]*t_lossz).sum()
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            #batchx, titer1 = sample(titer1, test_loader1)
            #datax = batchx[0].to(args.device)
            #batchy, titer2 = sample(titer2, test_loader2)
            #datay = batchy[0].to(args.device)
            evaluate(args.visualiser, data, datay, dataz, dataw, generator, 'x', args.device)
            args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss X')
            args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss Y')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
