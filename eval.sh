#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00
#SBATCH --job-name=longvideobench
#SBATCH --export=ALL
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
# set up python environment
conda activate llava

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,max_frames_num=32 \
    --tasks mlvu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs_frame/