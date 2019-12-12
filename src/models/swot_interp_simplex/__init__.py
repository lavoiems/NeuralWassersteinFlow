from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset1', type=str, default='visda')
    parser.add_argument('--dataset2', type=str, default='visda')
    parser.add_argument('--dataset-loc1', type=str, default='./data')
    parser.add_argument('--dataset-loc2', type=str, default='./data')
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
    dataset1 = getattr(images, args.dataset1)
    train_loader1, test_loader1, shape1, _ = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size)
    args.loaders1 = (train_loader1, test_loader1)
    args.shape1 = shape1

    dataset2 = getattr(images, args.dataset2)
    train_loader2, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size)
    args.loaders2 = (train_loader2, test_loader2)
    args.shape2 = shape2

    args.loader3 = (train_loader2, test_loader2)
    args.loader4 = (train_loader2, test_loader2)

    train(args)
