#!/bin/bash
set +o posix
echo 'Iterative pruning.'

exp="iter"
prn_epochs=10
tune_per_prn=5
ft_epochs=90
fixed_lr_epochs=10
bsz=64

seeds=(0)
datasets=("cifar100")
models=("vgg16_bn")
sparsities=(0.02)
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

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "SNIP" --itr_lr 1e-4 --lr 1e-2 --free_conv1

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "GraSP" --itr_lr 1e-4 --lr 1e-2 --free_conv1

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "SynFlow" --itr_lr 1e-4 --lr 1e-2 --free_conv1

        done
      done
    done
  done
done

seeds=(0)
datasets=("cifar100")
models=("resnet20" "wrn20" "vgg16_bn")
sparsities=(0.035 0.007)
prn_scopes=("global" "local")

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

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "Mag" --itr_lr 1e-4 --lr 1e-2 --free_conv1

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "SNIP" --itr_lr 1e-4 --lr 1e-2 --free_conv1

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "GraSP" --itr_lr 1e-4 --lr 1e-2 --free_conv1

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "SynFlow" --itr_lr 1e-4 --lr 1e-2 --free_conv1

        done
      done
    done
  done
done

models=("resnet20" "wrn20" "vgg16_bn")
sparsities=(0.05 0.1)
prn_scopes=("global" "local")

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

          python main.py --exp "$exp" --seed "$seed" --data "$data" --bsz "$bsz" \
          --model "$model" --sparsity "$sparsity" --prn_scope "$prn_scope" \
          --prn_epochs "$prn_epochs" --tune_per_prn "$tune_per_prn" --ft_epochs "$ft_epochs" --fixed_lr_epochs "$fixed_lr_epochs" \
          --pruner "GraSP" --itr_lr 1e-4 --lr 1e-2 --free_conv1

        done
      done
    done
  done
done
