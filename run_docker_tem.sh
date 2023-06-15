#!/bin/bash

CODE_DIR='/home/zhanfeng/Pruning/sparsity-indexed-ode/code'
MODEL_DIR='/home/zhanfeng/Models'
RESULTS_DIR='/home/zhanfeng/Pruning/sparsity-indexed-ode/results'
DATA_DIR='/home/zhanfeng/Data'

PORT='8891'

docker run -d --rm -h sparse-docker \
  -v $DATA_DIR:/HappyResearch/Data:ro -v $MODEL_DIR:/HappyResearch/Models -v $RESULTS_DIR:/HappyResearch/Results -v $CODE_DIR:/HappyResearch/code:ro \
  -u $(id -u):$(id -g) --shm-size=64gb --gpus '0' spode:1.0.1 /bin/bash -c "cd code && bash Scripts/cifar100/oneshot_ode_polarized_all.sh "

docker run -d --rm -h sparse-docker  \
-v $DATA_DIR:/HappyResearch/Data:ro -v $MODEL_DIR:/HappyResearch/Models -v $RESULTS_DIR:/HappyResearch/Results -v $CODE_DIR:/HappyResearch/code:ro \
-u $(id -u):$(id -g) --shm-size=64gb --gpus '1' spode:1.0.1 /bin/bash -c "cd code && bash Scripts/cifar10/oneshot_ode_polarized_all.sh "
