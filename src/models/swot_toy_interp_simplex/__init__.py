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
    dataset1 = toy.shapenet(args.train_batch_size, 0)
    dataset2 = toy.shapenet(args.train_batch_size, 1)
    dataset3 = toy.shapenet(args.train_batch_size, 2)
    dataset4 = toy.shapenet(args.train_batch_size, 3)
    tdataset1 = toy.shapenet(args.test_batch_size, 0)
    tdataset2 = toy.shapenet(args.test_batch_size, 1)
    tdataset3 = toy.shapenet(args.test_batch_size, 2)
    tdataset4 = toy.shapenet(args.test_batch_size, 3)
    args.loaders1 = (dataset1, tdataset1)
    args.loaders2 = (dataset2, tdataset2)
    args.loaders3 = (dataset3, tdataset3)
    args.loaders4 = (dataset4, tdataset4)
    args.shape1 = 3

    train(args)
