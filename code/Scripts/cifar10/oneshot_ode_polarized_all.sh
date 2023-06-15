#!/bin/bash
set +o posix
echo 'One-shot all pruning. seed = (1 2)'

exp="oneshot"
cuda_idx=0
ft_epochs=100
fixed_lr_epochs=10
bsz=64

seeds=(1 2)
datasets=("cifar10")
models=("resnet20" "wrn20" "vgg16_bn")
sparsities=(0.1 0.07 0.05 0.035 0.02)
prn_scopes=("global")

# Structural

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
            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
            --lr 1e-2 --free_conv1 \
            --mask_proc_option "ohh" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp

          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 \
          #            --mask_proc_option "ohm" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 \
          #            --mask_proc_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 \
          #            --mask_proc_option "qt" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 \
          #            --mask_proc_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "gau" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 \
          #            --mask_proc_option "qt" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp

        done
      done
    done
  done
done

# Unstructural

exp="oneshot"
seeds=(1 2)
datasets=("cifar10")
models=("resnet20" "wrn20" "vgg16_bn")
sparsities=(0.28 0.2 0.14 0.1 0.07)
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
            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
            --lr 1e-2 --free_conv1 --structural --mask_dim 1 \
            --mask_proc_option "ohh" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp

          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 --structural --mask_dim 1 \
          #            --mask_proc_option "ohm" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 --structural --mask_dim 1 \
          #            --mask_proc_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 --structural --mask_dim 1 \
          #            --mask_proc_option "qt" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 --structural --mask_dim 1 \
          #            --mask_proc_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "gau" --mask_proc_mxp
          #
          #          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          #            --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          #            --pruner "ODE" --prn_epochs 1 --N 100 --mom 0.0 --r 1.1 \
          #            --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          #            --lr 1e-2 --free_conv1 --structural --mask_dim 1 --structural --mask_dim 1 \
          #            --mask_proc_option "qt" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_score_option "Id" --mask_proc_mxp

        done
      done
    done
  done
done
