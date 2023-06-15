#!/bin/bash
set +o posix
echo 'Pretrain teacher models'

cuda_idx=1
seeds=(0)
datasets=("cifar10" "cifar100")
models=("resnet20" "vgg16_bn" "wrn20")

#datasets=("tiny_imagenet")
#models=("resnet50 vgg19_bn wrn34")

for ((s = 0; s < ${#seeds[@]}; s++)); do
  for ((d = 0; d < ${#datasets[@]}; d++)); do
    for ((i = 0; i < ${#models[@]}; i++)); do

      seed="${seeds[$s]}"
      data="${datasets[$d]}"
      model="${models[$i]}"

      python main.py --cuda_idx "$cuda_idx" --num_workers 12 \
        --seed "$seed" --exp "pretrain" \
        --data "$data" --model "$model"

      # python main.py --cuda_idx "$cuda_idx" --num_workers 8 \
      #   --seed "$seed" --exp "pretrain" \
      #   --data "$data" --model "$model" \
      #   --lr 1e-5 --pre_epochs 200 --bsz 128

    done

  done
done
