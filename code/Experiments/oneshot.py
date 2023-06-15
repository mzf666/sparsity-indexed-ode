import numpy as np
import pandas as pd
import torch

from Methods.utils import free_modules, init_masks_hooks
from Utils import load, routine, datasets


def run(args):
    '''
        Load data & model
    '''
    model = load.load_model(args.model_dir, args.model, args.data, args.device, args.pretrained)
    init_masks_hooks(model, args.structural, args.mask_dim)
    free_modules(model, args.free_bn, args.free_Id, args.free_bias, args.free_conv1)
    train_loader, test_loader = load.load_data(args.data_dir, args.data, args.bsz, args.num_workers, None)
    criterion = torch.nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.load_optimizer(args.opt)

    '''
        One-shot Pruning
    '''
    print(f'\nOne-shot pruning on {args.data} with {args.pruner} for {args.prn_epochs} epochs.\n')
    pruner = load.load_pruner(model, args.pruner, args.structural, args.mask_dim, **args.prn_kwargs)

    train_func = load.load_train_function(args.data, args.model)
    evaluate_func = load.load_evaluate_function(args.data, args.model, args.data_dir)

    sparsity = load.load_sparsity(model, args.sparsity, args.prn_scope)
    sparsity_schedule = load.load_sparsity_schedule(sparsity, args.prn_epochs, args.prn_schedule)

    results = []
    columns = ['sparsity', 'train_loss', 'train_acc1', 'train_acc5', 'val_loss', 'val_acc1', 'val_acc5']

    sparsity_tmp = 1.0
    val_loss, val_acc1, val_acc5 = evaluate_func(model, test_loader, args.device)

    row = [sparsity_tmp, np.nan, np.nan, np.nan, val_loss, val_acc1, val_acc5]
    results.append(row)
    if args.retrain_path is not None:
        args.prn_epochs = 0
        ckpt_path = args.retrain_path
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'{args.retrain_path} loaded.')

    for epoch in range(args.prn_epochs):
        ## Prune phase ##

        print(f'\nOne-shot prune Epoch [{epoch + 1}/{args.prn_epochs}], BEFORE prune = {sparsity_tmp * 100: .2f}%.')
        prn_bsz = 512 if args.data not in ['imagenet', 'tiny_imagenet'] else 256
        prune_loader = datasets.dataloader(args.data_dir, args.data, prn_bsz, True, args.num_workers, None)
        print(f'Prune loader batch size = {prn_bsz}.')

        print(f'Ideal sparsity = {sparsity_schedule[epoch]}.')
        routine.prune(model, pruner, criterion, prune_loader, sparsity_schedule[epoch], args.device, train=True)

        remainings, total, sparsity_tmp = pruner.stats(model)
        print(f'AFTER prune = {sparsity_tmp * 100 : .2f}% [{remainings}/{total}].')

        val_loss, val_acc1, val_acc5 = evaluate_func(model, test_loader, args.device)
        row = [sparsity_tmp, np.nan, np.nan, np.nan, val_loss, val_acc1, val_acc5]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{args.save_dir}/results.csv')

    if args.save_ckpt and args.retrain_path is None:
        optimizer, scheduler = None, None
        save_path = f'{args.save_dir}/model_pruned.pt'
        routine.save_dict(save_path, epoch + 1, sparsity_tmp, model, optimizer, scheduler, val_acc1, val_acc5, args)

    '''
        Final fine-tune
    '''
    print(f'\nFinal fine-tune for {args.ft_epochs} epochs.\n')
    best_acc = 0.

    opt_kwargs['weight_decay'] = args.weight_decay
    if args.pruner in ['ODE'] and args.mask_option == 'mask':
        print(
            f'Warning: using warmup lr schedule, T_warmup = {args.warmup_epochs}, lr {args.cold_start_lr} --> {args.lr}')
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = opt_class(optimizer_grouped_parameters, lr=args.lr, **opt_kwargs)
        scheduler = routine.WarmupCosineAnnealingLR(optimizer, T_warmup=args.warmup_epochs,
                                                    T_max=args.ft_epochs - args.fixed_lr_epochs,
                                                    eta_max=args.lr, eta_min=1e-6)

    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = opt_class(optimizer_grouped_parameters, lr=0.0, **opt_kwargs)
        scheduler = routine.WarmupCosineAnnealingLR(optimizer, T_warmup=args.warmup_epochs,
                                                    T_max=args.ft_epochs - args.fixed_lr_epochs,
                                                    eta_max=args.lr, eta_min=1e-6)
    if args.data == 'imagenet':
        no_decay = []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = opt_class(optimizer_grouped_parameters, lr=0.0, momentum=0.875, weight_decay=args.weight_decay)
        scheduler = routine.WarmupCosineAnnealingLR(optimizer, T_warmup=args.warmup_epochs,
                                                    T_max=args.ft_epochs - args.fixed_lr_epochs,
                                                    eta_max=args.lr, eta_min=1e-12)

    if args.data == 'imagenet':
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    for epoch in range(args.ft_epochs):
        train_loss, train_acc1, train_acc5 = train_func(model, optimizer, train_loader, args.device,
                                                        epoch)
        if epoch + 1 <= args.ft_epochs - args.fixed_lr_epochs:
            scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate_func(model, test_loader, args.device)
        _, _, sparsity_tmp = pruner.stats(model)
        print(f'[{epoch + 1}/{args.ft_epochs}] Current sparsity = {sparsity_tmp * 100 : .2f}%.')

        row = [sparsity_tmp, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{args.save_dir}/results.csv')

        if val_acc1 >= best_acc and args.save_ckpt:
            best_acc = val_acc1
            save_path = f'{args.save_dir}/model_tuned.pt'
            routine.save_dict(save_path, epoch + 1, sparsity_tmp, model, optimizer, scheduler, val_acc1, val_acc5, args)
