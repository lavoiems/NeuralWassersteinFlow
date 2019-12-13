import time
import torch
from torch import optim
from torch.distributions.dirichlet import Dirichlet
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

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
    criticz1 = model.Critic(shape1, **parameters)
    criticz2 = model.Critic(shape1, **parameters)
    criticw1 = model.Critic(shape1, **parameters)
    criticw2 = model.Critic(shape1, **parameters)
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
        'criticx1': criticx1,
        'criticx2': criticx2,
        'criticy1': criticy1,
        'criticy2': criticy2,
        'criticz1': criticz1,
        'criticz2': criticz2,
        'criticw1': criticw1,
        'criticw2': criticw2,
    }


@torch.no_grad()
def evaluate(visualiser, data, target, target2, generator, id, device):
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

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.scatter(*target2.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target3', f'{id}0')
    plt.clf()
    concentrations = [(1,0,0), (0,1,0), (0,0,1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.34, 0.33, 0.33)]
    for concentration in concentrations:
        plt.xlim(0,1)
        plt.ylim(0,1)
        t_ = torch.FloatTensor(concentration).to(device)
        t = torch.stack([t_] * data.shape[0])
        X = generator(data, t)
        plt.scatter(*X.cpu().numpy().transpose(), c=color_val)
        visualiser.matplotlib(fig, f'data{concentration}', f'{id}0')
        plt.clf()
    plt.close(fig)


@torch.no_grad()
def visualize_points(visualiser, fig, points, title, bound=1, c=None):
    #fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('off')

    if len(points) != 0:
        ax.scatter(*points, c=c)

    #ax.set_title('Points')
    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)
    ax.view_init(elev=-45, azim=80)
    visualiser.matplotlib(fig, title, None)
    plt.clf()


@torch.no_grad()
def evaluate_3d(visualiser, data, target, target2, target3, lx, ly, lz, lw, generator, id, device): 
    fig = plt.figure()
    visualize_points(visualiser, fig, data.cpu().numpy().transpose(), 'mu1', c=lx.cpu().numpy())
    visualize_points(visualiser, fig, target.cpu().numpy().transpose(), 'mu2', c=ly.cpu().numpy())
    visualize_points(visualiser, fig, target2.cpu().numpy().transpose(),'mu3', c=lz.cpu().numpy())
    visualize_points(visualiser, fig, target3.cpu().numpy().transpose(),'mu4', c=lw.cpu().numpy())

    concentrations = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (0.5,0.5,0,0), (0.5,0,0.5,0), (0.5,0,0,0.5), (0,0.5,0.5,0), (0,0.5,0,0.5), (0,0,0.5,0.5),
            (0.34, 0.33, 0.33,0), (0,0.34,0.33,0.33),(0.25,0.25,0.25,0.25)]
    concentration = [(1,0,0,0), (0.7,0.3,0,0), (0.5,0.5,0,0), (0.3,0.7,0,0),
                     (0.7,0,0.3,0), (0.6,0.2,0.2,0), (0.34,0.3,0.3,0), (0.2,0.6,0.2,0),(0,0.7,0.3,0),
                     (0.5,0,0.5,0), ()]
    for concentration in concentrations:
        t_ = torch.FloatTensor(concentration).to(device)
        t = torch.stack([t_] * data.shape[0])
        X = generator(data, t)
        visualize_points(visualiser, fig, X.cpu().numpy().transpose(), f'generated{concentration}', c=lx.cpu().numpy())
        plt.clf()
    plt.close(fig)


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
    criticw1 = models['criticw1'].to(args.device)
    criticw2 = models['criticw2'].to(args.device)
    print(generator)
    print(criticx1)
    print(criticy1)
    print(criticz1)
    print(criticw1)

    optim_criticx1 = optim.Adam(criticx1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticx2 = optim.Adam(criticx2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy1 = optim.Adam(criticy1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticy2 = optim.Adam(criticy2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticz1 = optim.Adam(criticz1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticz2 = optim.Adam(criticz2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticw1 = optim.Adam(criticw1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_criticw2 = optim.Adam(criticw2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2, iter3, iter4 = iter(train_loader1), iter(train_loader2), iter(train_loader3), iter(train_loader4)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2, titer3, titer4 = iter(test_loader1), iter(test_loader2), iter(test_loader3), iter(test_loader4)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        criticx1.train()
        criticx2.train()
        criticy1.train()
        criticy2.train()
        criticz1.train()
        criticz2.train()
        criticw1.train()
        criticw2.train()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            data = batchx[0].to(args.device)
            batchx, iter1 = sample(iter1, train_loader1)
            input_data = batchx[0].to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy[0].to(args.device)
            batchz, iter3 = sample(iter3, train_loader3)
            dataz = batchz[0].to(args.device)
            batchw, iter4 = sample(iter4, train_loader4)
            dataw = batchw[0].to(args.device)

            optim_criticx1.zero_grad()
            optim_criticx2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(input_data, data, args.eps, args.lp, criticx1, criticx2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticx1.step()
            optim_criticx2.step()

            optim_criticy1.zero_grad()
            optim_criticy2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(input_data, datay, args.eps, args.lp, criticy1, criticy2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticy1.step()
            optim_criticy2.step()

            optim_criticz1.zero_grad()
            optim_criticz2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(input_data, dataz, args.eps, args.lp, criticz1, criticz2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticz1.step()
            optim_criticz2.step()

            optim_criticw1.zero_grad()
            optim_criticw2.zero_grad()
            r_loss, g_loss, p = disc_loss_generation(input_data, dataw, args.eps, args.lp, criticw1, criticw2)
            (r_loss + g_loss + p).backward(mone)
            optim_criticw1.step()
            optim_criticw2.step()

        optim_generator.zero_grad()
        t_ = Dirichlet(torch.FloatTensor([1.,1.,1.,1.])).sample().to(args.device)
        t = torch.stack([t_]*input_data.shape[0])
        tinputdata = torch.cat([input_data]*args.nt)
        tdata = torch.cat([data]*args.nt)
        tdatay = torch.cat([datay]*args.nt)
        tdataz = torch.cat([dataz]*args.nt)
        tdataw = torch.cat([dataw]*args.nt)
        t_lossx = transfer_loss(tinputdata, tdata, args.nt, t, args.eps, args.lp, criticx1, criticx2, generator)
        t_lossy = transfer_loss(tinputdata, tdatay, args.nt, t, args.eps, args.lp, criticy1, criticy2, generator)
        t_lossz = transfer_loss(tinputdata, tdataz, args.nt, t, args.eps, args.lp, criticz1, criticz2, generator)
        t_lossw = transfer_loss(tinputdata, tdataw, args.nt, t, args.eps, args.lp, criticw1, criticw2, generator)
        t_loss = (t_[0]*t_lossx + t_[1]*t_lossy + t_[2]*t_lossz + t_[3]*t_lossw).sum()
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx[0].to(args.device)
            labelx = batchx[1].to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy[0].to(args.device)
            labely = batchy[1].to(args.device)
            batchz, titer3 = sample(titer3, test_loader3)
            dataz = batchz[0].to(args.device)
            labelz = batchz[1].to(args.device)
            batchw, titerw = sample(titer4, test_loader4)
            dataw = batchw[0].to(args.device)
            labelw = batchw[1].to(args.device)
            evaluate_3d(args.visualiser, datax, datay, dataz, dataw, labelx, labely, labelz, labelw, generator, 'x', args.device)
            d_loss = (r_loss+g_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Critic loss Y')
            args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss X')
            args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss Y')
            args.visualiser.plot(step=i, data=t_lossw.detach().cpu().numpy(), title=f'Generator loss W')
            args.visualiser.plot(step=i, data=p.detach().cpu().numpy(), title=f'Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
