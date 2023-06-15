#!/bin/bash

CODE_DIR='/home/zhanfeng/Pruning/sparsity-indexed-ode/code'
MODEL_DIR='/home/zhanfeng/Models'
RESULTS_DIR='/home/zhanfeng/Pruning/sparsity-indexed-ode/results'
DATA_DIR='/home/zhanfeng/Data'

PORT='8891'

docker run -it --rm -h sparse-docker \
  -v $DATA_DIR:/HappyResearch/Data:ro \
  -v $MODEL_DIR:/HappyResearch/Models \
  -v $RESULTS_DIR:/HappyResearch/Results \
  -v $CODE_DIR:/HappyResearch/code:ro \
  -u $(id -u):$(id -g) --shm-size=64gb --gpus '0' \
  spode:1.0.1 /bin/bash -c "cd code && bash Scripts/cifar100/summary.sh "

#docker run -it --rm -h sparse-docker \
#  -v $DATA_DIR:/HappyResearch/Data:ro \
#  -v $MODEL_DIR:/HappyResearch/Models \
#  -v $RESULTS_DIR:/HappyResearch/Results \
#  -v $CODE_DIR:/HappyResearch/code:ro \
#  -u $(id -u):$(id -g) --shm-size=64gb --gpus '1' \
#  spode:1.0.1 /bin/bash -c "cd code && bash Scripts/cifar100/summary.sh "

# '/HappyResearch/Results/oneshot_cifar10/wrn20_sp0.05/ODE_N100_r1.1_one_exp_fix_mom0.0_ohh_conv1_global/seed2/results.csv'
