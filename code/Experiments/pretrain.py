import os

import numpy as np
import pandas as pd
import torch
from Utils import load


def train_from_scratch_configs(args):
    exp_type = f'{args.data}_{args.model}'

    ## CIFAR-10 / 100 ##
    if exp_type in ['cifar10_vgg11', 'cifar100_vgg11']:
        args.opt = 'momentum'
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ['cifar10_vgg16', 'cifar100_vgg16',
                      'cifar10_vgg16_bn', 'cifar100_vgg16_bn',
                      'cifar10_vgg19_bn', 'cifar100_vgg19_bn']:
        args.opt = 'momentum'
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ['cifar10_resnet20', 'cifar100_resnet20',
                      'cifar10_resnet56', 'cifar100_resnet56', 'cifar100_mobilenet_v2']:
        args.opt = 'momentum'
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4
    elif exp_type in ['cifar100_vitb16', 'cifar10_vitb16', 'tiny_imagenet_vitb16']:
        args.opt = 'adamw'
        args.pre_epochs = 4
        args.bsz = 8
        args.lr = 2e-5
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4
    elif exp_type in ['cifar10_wrn20', 'cifar100_wrn20']:
        args.opt = 'momentum'
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    ## Tiny-ImagNet ##

    elif exp_type in ['tiny_imagenet_vgg19_bn']:
        args.opt = 'momentum'
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ['tiny_imagenet_resnet50']:
        args.opt = 'momentum'
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ['tiny_imagenet_wrn34']:
        args.opt = 'momentum'
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = '60,120,160'
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    else:
        raise NotImplementedError

    args.lr_milestones = [int(i) for i in args.lr_milestones.split(',')]
    print(f'Updated to {exp_type} configuration.')

    return args


def run(args):
    save_dir = f'{args.model_dir}/pretrained/{args.data}'
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = f'{save_dir}/{args.model}.pt'

    '''
        Load data & model
    '''

    train_loader, test_loader = load.load_data(args.data_dir, args.data, args.bsz, args.num_workers, None)
    if args.pretrained:
        print('Warning: Finetuning pretrained model ...')
        args = train_from_scratch_configs(args)
    else:
        print('Warning: Training model from scratch ...')
        args = train_from_scratch_configs(args)

    model = load.load_model(args.model_dir, args.model, args.data, args.device, args.pretrained)

    opt_class, opt_kwargs = load.load_optimizer(args.opt)

    '''
        Pretrain model
    '''
    print(f'\nPretrain model on {args.data} with {args.model} for {args.pre_epochs} epochs.\n')
    opt_kwargs['weight_decay'] = args.weight_decay
    opt_kwargs['weight_decay'] = args.weight_decay
    optimizer = opt_class(model.parameters(), lr=args.lr, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_epochs, eta_min=1e-6)

    train_func = load.load_train_function(args.data, args.model)
    evaluate_func = load.load_evaluate_function(args.data, args.model, args.data_dir)
    results = []
    columns = ['sparsity', 'lr', 'train_loss', 'train_acc1', 'train_acc5', 'val_loss', 'val_acc1', 'val_acc5']

    sparsity_tmp = 1.0
    val_loss, val_acc1, val_acc5 = evaluate_func(model, test_loader, args.device)
    row = [sparsity_tmp, optimizer.param_groups[0]['lr'], np.nan, np.nan, np.nan, val_loss, val_acc1, val_acc5]
    results.append(row)

    best_acc = 0.
    for epoch in range(args.pre_epochs):
        train_loss, train_acc1, train_acc5 = train_func(model, optimizer, train_loader, args.device,
                                                        epoch)
        scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate_func(model, test_loader, args.device)

        row = [sparsity_tmp, optimizer.param_groups[0]['lr'], train_loss, train_acc1, train_acc5, val_loss, val_acc1,
               val_acc5]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{save_dir}/{args.model}.csv')

        if val_acc1 >= best_acc:
            best_acc = val_acc1

            if os.path.exists(ckpt_path):
                existing_ckpt = torch.load(ckpt_path)
                existing_acc = existing_ckpt['acc']
                print(f'Current Acc = {best_acc: .2f}%.\n'
                      f'Existeing Acc = {existing_acc: .2f}%.')
                if best_acc < existing_acc:
                    print('Model skipped.\n')
                    continue

            ckpt = {
                'epoch': epoch,
                'acc': best_acc,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            print(f'Ckeckpoint at epoch [{epoch + 1}/{args.pre_epochs}] saved.\n'
                  f'Acc = {best_acc: .2f}%.\n')
