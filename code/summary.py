'''
    Model Selection with results.csv
'''
import argparse
import os
import socket
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


def build_args():
    HOST_NAME = socket.gethostname()
    DATA_DICT = {'sparse-docker': '/HappyResearch/Data', }
    MODEL_DICT = {'sparse-docker': "/HappyResearch/Models", }
    OUT_DICT = {'sparse-docker': "/HappyResearch/Results"}

    parser = argparse.ArgumentParser('Analysis results.')
    parser.add_argument('--out_dir', type=str, default=OUT_DICT[HOST_NAME])
    parser.add_argument('--exp', type=str, default='oneshot', help='oneshot | iter')
    parser.add_argument('--data', type=str, default='tiny_imagenet')
    parser.add_argument('--model', type=str, default='resnet50', help='vgg16_bn | wrn20 | resnet20')
    parser.add_argument('--sparsity', type=float, default=0.28)
    parser.add_argument('--pruners', type=str, default='Mag,SNIP,SynFlow,GraSP')
    parser.add_argument('--seed_list', type=str, default='0,1,2')

    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--prn_epochs', type=int, default=5)
    parser.add_argument('--tune_per_prn', type=int, default=5)
    parser.add_argument('--ft_epochs', type=int, default=120)
    parser.add_argument('--final_avg', type=int, default=5)

    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    print(f'Seed list = {args.seed_list}')

    ## Sparsity-ODE variants

    extra_pruners = []

    extra_pruners += [f'{name}_N{N}_r{r}_{mask_option}_{schedule}_{rt_schedule}_mom{mom}_{mask_proc_option}'
                      for name in ['ODE']
                      for N in ['100']
                      for r in ['1.01', '1.1']
                      for mask_option in ['one']
                      for schedule in ['exp']
                      for rt_schedule in ['fix']
                      for mom in ['0.0']
                      for mask_proc_option in ['ohh']]

    args.pruners = args.pruners.split(',')
    args.pruners += extra_pruners

    args.exp_name = f'{args.exp}_{args.data}/{args.model}_sp{args.sparsity}'
    args.pruners = [f'{prn}{description}_{scope}'
                    for prn in args.pruners
                    for description in ['', '_conv1', '_conv1_struct1']
                    for scope in ['local', 'global']]
    args.seed_list = [int(s) for s in args.seed_list.split(',')]
    args.epoch_threshold = args.pre_epochs + (args.prn_epochs * args.tune_per_prn)
    assert args.final_avg < args.ft_epochs, 'Set final_avg < final tune epochs.'

    return args


def merge_results(path, sheet_names, dfs):
    if os.path.exists(path):
        file_name = path.split('/')[-1]
        print(f'\tWarning: {file_name} exists. Skip to avoid over-writing!')
        pass
    else:
        with pd.ExcelWriter(path) as writer:
            for name, df in zip(sheet_names, dfs):
                # name = name.replace('[', '').replace(']', '')
                df.to_excel(writer, sheet_name=name, index=False)

            writer.save()


def summarize(pruner, args, min_length=70):
    seed_paths = [f'{args.out_dir}/{args.exp_name}/{pruner}/seed{seed}/results.csv'
                  for seed in args.seed_list]
    seed_exists = [seed for seed in args.seed_list
                   if os.path.exists(f'{args.out_dir}/{args.exp_name}/{pruner}/seed{seed}/results.csv')
                   and len(
            pd.read_csv(f'{args.out_dir}/{args.exp_name}/{pruner}/seed{seed}/results.csv', index_col=0)) >= min_length
                   #        and len(
                   # pd.read_csv(f'{args.out_dir}/{args.exp_name}/{pruner}/seed{seed}/results.csv', index_col=0)) <= (
                   #                    min_length + 5)
                   ]
    print(f'Seed paths = {seed_paths}')
    print(f'Seed exists = {seed_exists}')
    seed_dfs = [pd.read_csv(path, index_col=0)
                for path in seed_paths if os.path.exists(path)
                if len(pd.read_csv(path, index_col=0)) >= min_length
                # and len(pd.read_csv(path, index_col=0)) <= (min_length + 5)
                ]
    if len(seed_dfs) == 0:
        # print(f'Pruner {pruner} results does not exists.')
        return None

    summary = []
    columns = seed_dfs[0].keys()
    for seed, seed_df in zip(seed_exists, seed_dfs):
        row = []
        for col in columns:
            avg_val = seed_df[col][- args.final_avg:].mean()
            row.append(avg_val)
        summary.append(row)
    summary_df = pd.DataFrame(data=np.array(summary), columns=columns, index=seed_exists)

    ## Add mean & std to first row ##
    summary_mat = np.array(summary_df.values)
    mean_mat = summary_mat.mean(axis=0)
    std_mat = summary_mat.std(axis=0)
    first_row = np.zeros_like(mean_mat).tolist()
    for i in range(len(first_row)):
        first_row[i] = f'{mean_mat[i]: .4f} ({std_mat[i]: .4f})'
    first_row = pd.DataFrame(data=[first_row], columns=columns, index=['stat'])

    summary_df = pd.concat([first_row, summary_df], axis=0)

    dfs = [summary_df] + seed_dfs
    sheet_names = ['summary'] + [f'{pruner}_seed{seed}' for seed in args.seed_list]
    summary_path = f'{args.out_dir}/{args.exp_name}/{pruner}.xlsx'
    merge_results(summary_path, sheet_names, dfs)

    print(f'\t {pruner} seed {seed_exists} summarized successfully.')

    return first_row


if __name__ == '__main__':
    warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))
    args = build_args()
    print(f'\n\nWarning: summarizing {args.exp} {args.data} {args.model} sp = {args.sparsity} ...\n\n')
    EPOCH_THRESHOLD = args.epoch_threshold
    # args.out_dir = f'{args.out_dir}/old_results_masked_bias'

    '''
        1. Summary for each pruner
    '''
    all_summary_rows = []
    exists_pruners = []
    for pruner in args.pruners:
        print(f'pruner = {pruner}')
        row = summarize(pruner, args)
        if row is None:
            continue
        all_summary_rows.append(row)
        exists_pruners.append(pruner)

    all_summary_rows = pd.concat(all_summary_rows, axis=0, ignore_index=True)
    all_summary_rows.index = exists_pruners
    path = f'{args.out_dir}/{args.exp_name}.csv'
    all_summary_rows.to_csv(path)
    print('\n\nTable summarized successfully. \n\n')

    '''
        2. Plot learning curves
    '''
    if args.plot:
        fig_path = f'{args.out_dir}/{args.exp_name}.pdf'
        keys = ['train_acc1', 'train_loss',
                'val_acc1', 'val_loss']
        key_dfs = []

        for key in keys:
            key_df = []
            for seed in args.seed_list:
                for pruner in exists_pruners:
                    path = f'{args.out_dir}/{args.exp_name}/{pruner}/seed{seed}/results.csv'
                    if os.path.exists(path):
                        df = pd.read_csv(path, index_col=0)
                        df = df[[key]]
                        df['seed'] = seed
                        df['pruner'] = pruner
                        df['epochs'] = list(range(len(df[key])))
                        key_df.append(df)

            key_df = pd.concat(key_df, axis=0, ignore_index=True)
            key_dfs.append(key_df)

        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols + 1, figsize=(12, 8),
                                 gridspec_kw=dict(width_ratios=[4] * ncols + [2],
                                                  height_ratios=[4] * nrows))
        for r in range(nrows):
            axes[r, -1].remove()
        fig.subplots_adjust(hspace=0.3, wspace=0.25)

        for idx, (key, df) in enumerate(zip(keys, key_dfs)):
            r = idx // ncols
            c = idx % ncols
            legend = 'auto' if (c + 1) == ncols else False

            p = sns.lineplot(data=df, x='epochs', y=key, ax=axes[r, c],
                             hue='pruner', style='pruner',
                             err_style="bars", errorbar=('ci', 68), legend=legend)
            p.set_title(key)
            p.set_xlabel(None)
            p.set_ylabel(None)
            # p.set_ylim((0.5,1))
            # if (r + 1) != nrows:
            #     p.set_xlabel(None)
            # if c != 0:
            #     p.set_ylabel(None)
            if (c + 1) == ncols:
                sns.move_legend(axes[r, c], "upper left", bbox_to_anchor=(1.06, 1), title=None)

        plt.suptitle(f'{args.exp} {args.data} {args.model} sp={args.sparsity}', fontsize=18)
        plt.savefig(fig_path, dpi=300)
        plt.cla()
        print('Plot finished successfully. \n\n')

    else:
        print('Warning: plot skipped. \n\n')
