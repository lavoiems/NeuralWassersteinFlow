from .train import train
from common.loaders import toy


def parse_args(parser):
    parser.add_argument('--dataset1', type=str, default='pointcloud')
    parser.add_argument('--dataset2', type=str, default='pointcloud')
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--nt', type=float, default=2)
    parser.add_argument('--lp', type=float, default=2)


def execute(args):
    print(args)
    #dataset1 = toy.spiral(1, args.train_batch_size, translationx=0, translationy=0)
    #dataset2 = toy.spiral(-1, args.train_batch_size, translationx=0, translationy=0)
    dataset1 = toy.gaussian2(args.train_batch_size)
    dataset2 = toy.mixture2(args.train_batch_size)
    args.loaders1 = (dataset1, dataset1)
    args.loaders2 = (dataset2, dataset2)
    args.shape1 = 2
    args.shape2 = 2

    train(args)
