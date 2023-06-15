#!/bin/bash
set +o posix
echo 'Iterative pruning.'

exp="iter"
cuda_idx=0
prn_epochs=10
tune_per_prn=5
ft_epochs=90
fixed_lr_epochs=10
bsz=64

seeds=(0)
datasets=("cifar100")
sparsities=(0.14)
models=("wrn20")
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

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "qt" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gau" --mask_proc_score_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gau" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

        done
      done
    done
  done
done

seeds=(0)
datasets=("cifar100")
sparsities=(0.28)
models=("vgg16_bn" "resnet20" "wrn20")
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

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "ohm" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "qt" --mask_proc_score_option "qt" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "qt" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gau" --mask_proc_score_option "gau" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gau" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

        done
      done
    done
  done
done

seeds=(0)
datasets=("cifar100")
sparsities=(0.07 0.14 0.28)
models=("vgg16_bn" "resnet20" "wrn20")
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

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "qtm" --mask_proc_score_option "qtm" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "qtm" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gaum" --mask_proc_score_option "gaum" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

          python main.py --exp "$exp" --seed "$seed" --cuda_idx "$cuda_idx" --num_workers 8 --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "ODE" --N 100 --mom 0.0 --r 1.1 \
          --itr_lr 1e-2 --lr 3e-3 --free_conv1 --structural --mask_dim 1 \
          --score_option "mp" --mask_option "one" --sparsity_option "l2" --schedule "exp" \
          --mask_proc_option "gaum" --mask_proc_score_option "Id" --mask_proc_eps 0.9 --mask_proc_ratio 0.9 --mask_proc_mxp

        done
      done
    done
  done
done
