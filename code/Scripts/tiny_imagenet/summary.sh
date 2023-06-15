#!/bin/bash
set +o posix
echo 'Summarizing experiment results ... '

data="tiny_imagenet"
pruners="Mag,MagRand,SNIP,SynFlow,GraSP"
models=("resnet50" "wrn34" "vgg19_bn")
sparsitys=(0.28 0.2 0.14 0.1 0.07)
#models=("resnet50")
#sparsitys=(0.05)

for ((i = 0; i < ${#models[@]}; i++)); do
  for ((j = 0; j < ${#sparsitys[@]}; j++)); do

    model="${models[$i]}"
    sparsity="${sparsitys[$j]}"

    echo -e "\n\n Oneshot: Model = $model, sparsity = $sparsity.\n\n"

    python summary.py --exp "oneshot" --data "$data" --pruners "$pruners" \
      --sparsity "$sparsity" --model "$model"

    #    echo -e "\n\n Iterative: Model = $model, sparsity = $sparsity.\n\n"
    #
    #    python summary.py --exp "iter" --data "$data" --pruners "$pruners" \
    #      --sparsity "$sparsity" --model "$model"

  done
done

echo 'Merging experiment results ... '

to_dir="result_dir"

python merge_results.py --from_dir "result_dir/oneshot_tiny_imagenet" \
  --to_dir "$to_dir" --type csv

python merge_results.py --from_dir "result_dir/oneshot_tiny_imagenet" \
  --to_dir "$to_dir" --type pdf

python merge_results.py --from_dir "result_dir/iter_tiny_imagenet" \
  --to_dir "$to_dir" --type csv

python merge_results.py --from_dir "result_dir/iter_tiny_imagenet" \
  --to_dir "$to_dir" --type pdf
