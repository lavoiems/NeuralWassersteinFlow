from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset1', type=str, default='mnist')
    parser.add_argument('--h-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--d-updates', type=int, default=5)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--lp', type=float, default=1)


def execute(args):
    dataset1 = getattr(images, args.dataset1)
    train_loader1, test_loader1, shape1, _ = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size)
    args.loaders1 = (train_loader1, test_loader1)
    args.shape1 = shape1

    train(args)
