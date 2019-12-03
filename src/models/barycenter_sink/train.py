import time
import torch
from torch import optim
import matplotlib.pyplot as plt

from common.util import sample, save_models
from common.sinkhorn import sinkhorn_loss
from common.initialize import initialize, infer_iteration
from . import model


def define_models(shape1, **parameters):
    generator = model.Generator(shape1, **parameters)
    return {
        'generator': generator,
    }


@torch.no_grad()
def evaluate(visualiser, data, data1, target, generator, id):
    fig = plt.figure()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(*data1.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target1', f'{id}0')
    plt.clf()
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.scatter(*target.cpu().numpy().transpose())
    visualiser.matplotlib(fig, 'target2', f'{id}0')
    plt.clf()
    plt.xlim(0,1)
    plt.ylim(0,1)
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
    print(generator)

    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    titer1, titer2 = iter(test_loader1), iter(test_loader2)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        batchx, iter1 = sample(iter1, train_loader1)
        data = batchx.to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        datay = batchy.to(args.device)

        optim_generator.zero_grad()
        x = generator(data)
        t_lossx = sinkhorn_loss(x, data, args.eps, data.shape[0], 500, args.device, p=args.lp)**args.p_exp
        t_lossy = sinkhorn_loss(x, datay, args.eps, data.shape[0], 500, args.device, p=args.lp)**args.p_exp
        ((1-args.alphat)*t_lossx + args.alphat*t_lossy).backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            generator.eval()
            print('Iter: %s' % i, time.time() - t0)
            batchx, titer1 = sample(titer1, test_loader1)
            datax = batchx.to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            datay = batchy.to(args.device)
            evaluate(args.visualiser, datax, datax, datay, generator, 'x')
            args.visualiser.plot(step=i, data=t_lossx.detach().cpu().numpy(), title=f'Generator loss y')
            args.visualiser.plot(step=i, data=t_lossy.detach().cpu().numpy(), title=f'Generator loss x')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
