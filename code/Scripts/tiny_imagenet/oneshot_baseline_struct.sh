#!/bin/bash
set +o posix
echo 'One-shot pruning.'

exp="oneshot"
cuda_idx=0
ft_epochs=70
fixed_lr_epochs=10
bsz=128

seeds=(0)
datasets=("tiny_imagenet")
models=("resnet50" "vgg19_bn" "wrn34")
sparsities=(0.28 0.2 0.14 0.07)
prn_scopes=("local")

for ((s = 0; s < ${#seeds[@]}; s++)); do
  for ((d = 0; d < ${#datasets[@]}; d++)); do
    for ((c = 0; c < ${#sparsities[@]}; c++)); do
      for ((i = 0; i < ${#models[@]}; i++)); do
        for ((j = 0; j < ${#prn_scopes[@]}; j++)); do

          seed="${seeds[$s]}"
          data="${datasets[$d]}"
          sparsity="${sparsities[$c]}"
          model="${models[$i]}"
          prn_scope="${prn_scopes[$j]}"

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
            --pruner "SNIP" --prn_epochs 1 --lr 1e-3 --free_conv1 --structural --mask_dim 1

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
            --pruner "SynFlow" --prn_epochs 100 --lr 1e-3 --free_conv1 --structural --mask_dim 1

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
            --pruner "GraSP" --prn_epochs 1 --lr 1e-3 --free_conv1 --structural --mask_dim 1

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
            --pruner "Mag" --prn_epochs 1 --lr 1e-3 --free_conv1 --structural --mask_dim 1

        done
      done
    done
  done
done
