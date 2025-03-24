#!/bin/bash -l
#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=no_compression_addllm_sharegpt
#SBATCH --export=ALL
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load cuda/12.6
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export BNB_CUDA_VERSION=122
export ACCELERATE_DISABLE_NUMA_AFFINITY=1

module load python
conda activate llava
module load gcc/9.4.0
bash scripts/train/finetune_ov.sh