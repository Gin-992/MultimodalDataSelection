#!/bin/bash


# pip install wheel
# pip install packaging
# pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
# pip install -r train-qwen-requirements.txt

# sudo apt update
# sudo apt install ffmpeg


NPROC_PER_NODE=8
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=/opt/tiger/vlm/MultimodalDataSelection/qwen-vl-finetune/scripts/zero3.json

# Model configuration
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
# MODEL_PATH=/mnt/hdfs/andrew.estornell/vlm/Qwen2.5-VL-7B-Instruct
llm=$MODEL_PATH  # Using HuggingFace model ID

# Training hyperparameters
lr=$LR
batch_size=$BATCH_SIZE
grad_accum_steps=$GRAD_ACCUM

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
# DATASET='mmds_itqa'
#DATASET=mmds_mm
datasets=$DATASET

# Output configuration
run_name=$RUN_NAME
output_dir=./$RUN_NAME

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}