import argparse
import os
import socket

# import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import sys
import json
import time
import random
import warnings

import numpy as np
import torch
import torchvision

from Experiments import iter, oneshot, pretrain#, tune, track, tune_track
from Utils import misc


def randomize(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args():
    HOST_NAME = socket.gethostname()
    DATA_DICT = {'sparse-docker': '/HappyResearch/Data', }
    MODEL_DICT = {'sparse-docker': "/HappyResearch/Models", }
    OUT_DICT = {'sparse-docker': "/HappyResearch/Results"}

    parser = argparse.ArgumentParser(description='Pruning with CV data.')
    parser.add_argument('--exp', type=str, default='oneshot',
                        help='oneshot | iter | pretrain | tune | track | tune_track')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--seed', type=int, default=124)
    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default=DATA_DICT[HOST_NAME])
    parser.add_argument('--data', type=str, default='cifar100',
                        help='cifar10 | cifar100 | tiny_imagenet | imagenet')
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--model_dir', type=str, default=MODEL_DICT[HOST_NAME])
    parser.add_argument('--model', type=str, default='vgg19_bn', help='resnet20 | vgg16_bn | wrn20')
    parser.add_argument('--use_init', action='store_true', default=False)
    parser.add_argument('--use_baseline', action='store_true', default=False)

    parser.add_argument('--opt', type=str, default='momentum', help='sgd | momentum | adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--itr_lr', type=float, default=1e-4)
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    parser.add_argument('--lr_milestones', type=str, default='50,80,120')
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--pre_epochs', type=int, default=2)
    parser.add_argument('--prn_epochs', type=int, default=1)
    parser.add_argument('--tune_per_prn', type=int, default=2)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--fixed_lr_epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--cold_start_lr', type=float, default=1e-4)

    parser.add_argument('--pruner', type=str, default='ODE',
                        help='Rand | Mag | SNIP | SynFlow | GraSP | ODE | ODENaive')
    parser.add_argument('--structural', action='store_true', default=False)
    parser.add_argument('--mask_dim', type=int, default=1, help='for structural pruning: 0 = out | 1 = in')
    parser.add_argument('--prn_scope', type=str, default='global', help='local | global')
    parser.add_argument('--prn_schedule', type=str, default='exponential')
    parser.add_argument('--prn_dataset_ratio', type=float, default=0.3)
    parser.add_argument('--sparsity', type=float, default=0.02)
    parser.add_argument('--retrain_path', type=str, default=None)

    parser.add_argument('--free_bn', action='store_true', default=True)
    parser.add_argument('--free_Id', action='store_true', default=True)
    parser.add_argument('--free_bias', action='store_true', default=True)
    parser.add_argument('--free_conv1', action='store_true', default=True)

    # ========================== Sparsity-ODE hyper-parameters ======================== #
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--r', type=float, default=1.1)
    parser.add_argument('--ode_scope', type=str, default='global', help='global | local')
    parser.add_argument('--energy_option', type=str, default='CE', help='CE | P1 | CEP1')
    parser.add_argument('--sparsity_option', type=str, default='l2', help='l1 | l2 | l1p')
    parser.add_argument('--mask_option', type=str, default='one', help='Dont change')

    parser.add_argument('--mask_proc_option', type=str, default='ohh', help='Id | qt | oh | ohh | gau')
    parser.add_argument('--mask_proc_eps', type=float, default=0.9)
    parser.add_argument('--mask_proc_ratio', type=float, default=0.9)
    parser.add_argument('--mask_proc_score_option', type=str, default='Id', help='Dont change')
    parser.add_argument('--mask_proc_mxp', action='store_true', default=True, help='Dont change')

    ## For ODENaive
    parser.add_argument('--eq_constrain', action='store_true', default=False)
    parser.add_argument('--normalized', action='store_true', default=True, help='Dont change')
    parser.add_argument('--lam', type=float, default=10)

    parser.add_argument('--quant', action='store_true', default=False)

    # ================================================================================= #

    parser.add_argument('--out_dir', type=str, default=OUT_DICT[HOST_NAME])
    parser.add_argument('--ckpt_freq', type=int, default=300)

    parser.add_argument('--save_ckpt', action='store_true', default=False)
    parser.add_argument('--score_option', type=str, default='mp', help='Dont change')
    parser.add_argument('--schedule', type=str, default='hess', help='lin | exp | invexp | hess')
    parser.add_argument('--rt_schedule', type=str, default='fix', help='fix | invexp | hess | auto')
    parser.add_argument('--mom', type=float, default=0., help='momentum coeff for ODE score update')
    parser.add_argument('--theta_opt', type=str, default='plain',
                        help='plain | sgd | adam | momentum | plain_all | sgd_all | adam_all | momentum_all')
    parser.add_argument('--rho', type=float, default=1e-6, help='for ODE-sys: param step size = rho * dt')
    parser.add_argument('--inner_N', type=int, default=0)

    args = parser.parse_args()

    args.device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
    args.pretrained = not args.use_init
    args.lr_milestones = [int(i) for i in args.lr_milestones.split(',')]

    # ================================ Pruner Config ================================= #
    import yaml
    from easydict import EasyDict as edict
    if args.pruner in ['ODE', 'ODETrack', 'ODENaive', ] and args.exp in ['iter','oneshot']:
        config_path = f'./Methods/Configs/{args.pruner}.yaml'
        args.prn_kwargs = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
        args.prn_kwargs['N'] = args.N
        args.prn_kwargs['r'] = args.r
        args.prn_kwargs['ode_scope'] = args.ode_scope
        args.prn_kwargs['E'] = args.energy_option
        args.prn_kwargs['G'] = args.sparsity_option
        args.prn_kwargs['score_option'] = args.score_option
        args.prn_kwargs['mask_option'] = args.mask_option
        args.prn_kwargs['mask_proc_kwargs'] = {
            'mask_proc_option': args.mask_proc_option,
            'mask_proc_eps': args.mask_proc_eps,
            'mask_proc_ratio': args.mask_proc_ratio,
            'mask_proc_score_option': args.mask_proc_score_option,
            'mask_proc_mxp': args.mask_proc_mxp
        }
        args.prn_kwargs['schedule'] = args.schedule
        args.prn_kwargs['rt_schedule'] = args.rt_schedule
        args.prn_kwargs['momentum'] = args.mom

        ## for Naive ODE
        if args.pruner == 'ODENaive':
            args.prn_kwargs['eq_constrain'] = args.eq_constrain
            args.prn_kwargs['normalized'] = args.normalized
            args.prn_kwargs['lam'] = args.lam

        ## Write ODE description

        for key in args.prn_kwargs.keys():
            if key in ['N', 'r', 'momentum']:
                args.description += f'_{key[:3]}{args.prn_kwargs[key]}'
            if key in ['mask_option', 'schedule', 'rt_schedule']:
                args.description += f'_{args.prn_kwargs[key]}'

        args.description += f'_{args.mask_proc_option}'
        if args.mask_proc_option in ['qt', 'gau']:
            args.description += f'{args.mask_proc_eps}{args.mask_proc_ratio}'

        if args.pruner == 'ODENaive':
            args.description += f'lam{args.lam}'
            if args.eq_constrain:
                args.description += '_eq'

    elif args.pruner == 'REG':
        config_path = f'./Methods/Configs/{args.pruner}.yaml'
        args.prn_kwargs = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
        args.prn_kwargs['N'] = args.N
        args.prn_kwargs['r'] = args.r
        args.prn_kwargs['E'] = args.energy_option
        for key in args.prn_kwargs.keys():
            if key in ['N', 'r']:
                args.description += f'_{key[:3]}{args.prn_kwargs[key]}'

        args.description += f'_{args.mask_proc_option}'
    else:
        args.prn_kwargs = {}

    # assert args.free_bias and args.free_bn and args.free_Id, 'Warning: by convention, we do NOT prune bias, bn or identity!'
    if args.free_conv1:
        args.description += '_conv1'
    if args.structural:
        args.description += f'_struct{args.mask_dim}'

    # ================================================================================= #

    if args.exp == 'pretrain':
        args.save_dir = f'{args.model_dir}/pretrained/{args.data}'
    elif args.exp == 'oneshot':
        args.save_dir = f"{args.out_dir}/{args.exp}_{args.data}/{args.model}_sp{args.sparsity}/{args.pruner}{args.description}_{args.prn_scope}/seed{args.seed}"

    elif args.exp == 'tune_track':
        args.save_dir = f'{args.out_dir}/oneshot_{args.data}/{args.model}_sp{args.sparsity}/{args.pruner}{args.description}_{args.prn_scope}/seed{args.seed}'
    else:
        args.save_dir = f'{args.out_dir}/{args.exp}_{args.data}/{args.model}_sp{args.sparsity}/{args.pruner}{args.description}_{args.prn_scope}/seed{args.seed}'
    os.makedirs(args.save_dir, exist_ok=True)

    if args.pruner in ['ODE', 'ODENaive']:
        args.prn_kwargs['save_dir'] = args.save_dir

    time_stamp = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    sys.stdout = misc.Tee(f'{args.save_dir}/{time_stamp}_log.txt')
    sys.stderr = misc.Tee(f'{args.save_dir}/{time_stamp}_err.txt')

    print("Environment:")
    print("\tTime: {}".format(time_stamp))
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    with open(args.save_dir + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)

    if args.structural:
        args.sparsity = args.sparsity ** 0.5

    return args


if __name__ == '__main__':
    warnings.simplefilter("ignore", UserWarning)

    args = build_args()
    randomize(args.seed)

    if args.exp == 'iter':
        iter.run(args)
    elif args.exp == 'oneshot':
        oneshot.run(args)
    elif args.exp == 'pretrain':
        pretrain.run(args)
    else:
        raise NotImplementedError
