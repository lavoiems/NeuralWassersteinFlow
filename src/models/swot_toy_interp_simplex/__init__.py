from .train import train
from common.loaders import toy


def parse_args(parser):
    parser.add_argument('--dataset1', type=str, default='gaussian')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--lp', type=int, default=2)
    parser.add_argument('--nt', type=int, default=2)
    parser.add_argument('--p-exp', type=int, default=2)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)


def execute(args):
    print(args)
    #dataset1 = toy.gaussian2(args.train_batch_size)
    #dataset2 = toy.mixture2(args.train_batch_size)
    #tdataset1 = toy.gaussian2(args.test_batch_size)
    #tdataset2 = toy.mixture2(args.test_batch_size)
    #dataset1 = toy.spiral(1, args.train_batch_size, translationx=0, translationy=0)
    #dataset2 = toy.spiral(-1, args.train_batch_size, translationx=0, translationy=0)
    dataset1 = toy.png(args.train_batch_size, 'square.png')
    dataset2 = toy.png(args.train_batch_size, 'circle.png')
    dataset3 = toy.png(args.train_batch_size, 'star.png')
    tdataset1 = toy.png(args.test_batch_size, 'square.png')
    tdataset2 = toy.png(args.test_batch_size, 'circle.png')
    tdataset3 = toy.png(args.test_batch_size, 'star.png')
    args.loaders1 = (dataset1, tdataset1)
    args.loaders2 = (dataset2, tdataset2)
    args.loaders3 = (dataset3, tdataset3)
    args.shape1 = 2

    train(args)
