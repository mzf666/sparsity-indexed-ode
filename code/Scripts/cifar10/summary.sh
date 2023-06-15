#!/bin/bash
set +o posix
echo 'Summarizing experiment results ... '

data="cifar10"
pruners="Mag,MagRand,SNIP,SynFlow,GraSP"
models=("resnet20" "wrn20" "vgg16_bn")
#sparsitys=(0.02 0.035 0.05 0.07 0.1 0.14 0.2 0.28 0.5)
sparsitys=(0.05)

for ((i = 0; i < ${#models[@]}; i++)); do
  for ((j = 0; j < ${#sparsitys[@]}; j++)); do

    model="${models[$i]}"
    sparsity="${sparsitys[$j]}"

    python summary.py --exp "oneshot" --data "$data" --pruners "$pruners" \
      --sparsity "$sparsity" --model "$model" --plot

    #    python summary.py --exp "iter" --data "$data" --pruners "$pruners" \
    #      --sparsity "$sparsity" --model "$model" --plot

  done
done

echo 'Merging experiment results ... '

python merge_results.py --dataset "cifar10" --exp "oneshot" --type "csv"

python merge_results.py --dataset "cifar10" --exp "oneshot" --type "pdf"

#python merge_results.py --dataset "cifar10" --exp "iter" --type "csv"
#
#python merge_results.py --dataset "cifar10" --exp "iter" --type "pdf"

#to_dir="result_dir"
#
#python merge_results.py --from_dir "result_dir/oneshot_cifar10" \
#  --to_dir "$to_dir" --type csv
#
#python merge_results.py --from_dir "result_dir/oneshot_cifar10" \
#  --to_dir "$to_dir" --type pdf

#python merge_results.py --from_dir "result_dir/iter_cifar10" \
#  --to_dir "$to_dir" --type csv
#
#python merge_results.py --from_dir "result_dir/iter_cifar10" \
#  --to_dir "$to_dir" --type pdf
