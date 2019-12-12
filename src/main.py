import time
import argparse
import torch
import numpy as np
from common.util import set_paths, create_paths, dump_args
from models import barycenter_sink
from models import barycenter_swgan
from models import barycenter_wgan
from models import gan
from models import legendre_duality
from models import legendre_duality_interp
from models import legendre_duality_toy
from models import legendre_duality_toy_interp
from models import swgan
from models import swgan_interp
from models import swgan_toy
from models import swgan_toy_c
from models import swgan_toy_interp
from models import swot_sink_toy_interp
from models import swot_toy_interp
from models import swot_toy_interp_simplex
from models import swot_interp
from models import swot_interp_simplex
from models import swgan_interp_simplex

_models_ = {
    'barycenter_sink': barycenter_sink,
    'barycenter_swgan': barycenter_swgan,
    'barycenter_wgan': barycenter_wgan,
    'gan': gan,
    'legendre_duality': legendre_duality,
    'legendre_duality_interp': legendre_duality_interp,
    'legendre_duality_toy': legendre_duality_toy,
    'legendre_duality_toy_interp': legendre_duality_toy_interp,
    'swgan': swgan,
    'swgan_interp': swgan_interp,
    'swgan_toy': swgan_toy,
    'swgan_toy_c': swgan_toy_c,
    'swgan_toy_interp': swgan_toy_interp,
    'swot_sink_toy_interp': swot_sink_toy_interp,
    'swot_toy_interp': swot_toy_interp,
    'swot_toy_interp_simplex': swot_toy_interp_simplex,
    'swot_interp': swot_interp,
    'swot_interp_simplex': swot_interp_simplex,
    'swgan_interp_simplex': swgan_interp_simplex,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--run-id', type=str, default=str(time.time()))
    parser.add_argument('--root-path', default='./experiments/')
    parser.add_argument('--server', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--visdom_dir', type=str, default='.')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--evaluate', type=int, default=100)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    for name, model in _models_.items():
        model_parser = subparsers.add_parser(name)
        model.parse_args(model_parser)
        model_parser.set_defaults(func=model.execute)
        model_parser.set_defaults(model=name)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    args.run_name = f'{args.exp_name}_{args.run_id}-{args.seed}'
    set_paths(args)
    create_paths(args.save_path, args.model_path, args.log_path)
    dump_args(args)

    if args.visdom:
        from common.visualise import Visualiser
        args.visualiser = Visualiser(args.server, args.port, f'{args.exp_name}_{args.run_id}', args.reload, '.')
    else:
        from common.tensorboard import Visualiser
        args.visualiser = Visualiser(args.log_path, args.exp_name)
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    args.visualiser.args(args)
    args.func(args)
