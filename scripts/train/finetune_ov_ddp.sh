#!/bin/bash

# No need to modify CUDA_VISIBLE_DEVICES
# No need to set LOCAL_RANK

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=DEBUG
export USE_PYTORCH_KERNEL_CACHE=0
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=3600
export NCCL_P2P_DISABLE=1
export CUTLASS_PATH=/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/cutlass
export WANDB_API_KEY="638aa591e9881cd840eb171df3f625bcd7613d14"


NNODES=${SLURM_JOB_NUM_NODES}
NODE_RANK=${SLURM_NODEID}
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
MASTER_PORT=29510  # You can pick any free port

# <<< END IMPORTANT >>>

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-0.5b-qwen2_KIT_position_8tokens_adapter_GC"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/test.yaml \
    --image_folder /hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos \
    --video_folder /home/hk-project-p0022560/tum_tyz7686/llava-video/videos \
    --mm_tunable_parts="larimar_model,recurrent_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position one_token \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --memory_transformer_lr 1e-4 \
    --memory_key_value_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --force_sample False \
    --frames_upbound 250 \
    --attn_implementation "flash_attention_2"
