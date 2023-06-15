[comment]: <> (# sparsity-indexed-ode)
# Sparsity-indexed-ODE
**[ICML2023] Neural Pruning via Sparsity-indexed ODE: A Continuous Sparsity Viewpoint**<br>
Zhanfeng Mo<sup>1</sup>, Haosen Shi<sup>1,2</sup>, Sinno Jialin Pan<sup>1,3</sup><br>
<sup>1</sup> <sub>School of Computer Science and Engineering, Nanyang Technological University</sub><br />
<sup>2</sup> <sub>Continental-NTU Corporate Lab, Nanyang Technological University</sub><br /> 
<sup>3</sup> <sub>Department of Computer Science and Engineering, Chinese University of Hong Kong. </sub><br /> 


Official implementation of "Neural Pruning via Sparsity-indexed ODE: A Continuous Sparsity Viewpoint, ICML 2023".
## Environment
We provide a NVIDIA-Docker image to facilitate researchers in reproducing the results reported in our paper and conducting further studies using our code. 

1. Build docker image from our Dockerfile
```bash
bash ./build_docker_com.sh
```

2. Two scripts `run_docker_(summary/tem).sh` are provided for running the experiments and summarizing the results. 
   Change the dirname in these two scripts to your local dirname.
```bash
bash ./run_docker_tem.sh
```

3. More scripts are provided in `Scripts/(datasetname)/`

## Update
- upload the link of paper
- ~~15/06/2023 init readme and code~~

## Contact
If you have any questions about this work, please feel easy to contact us (ZHANFENG001 (AT) ntu.edu.sg).


## Thanks
This code is heavily borrowed from [[SynFlow]](https://github.com/ganguli-lab/Synaptic-Flow).

## Citation
If you use this code for your research, please cite our paper, "Neural Pruning via Sparsity-indexed ODE: A Continuous Sparsity Viewpoint".