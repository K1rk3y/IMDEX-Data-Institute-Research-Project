#!/bin/bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
JOB_NAME='videomamba_middle_f16_res224'
OUTPUT_DIR="./checkpoints/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='celebdf_dataset'

# Direct execution instead of using srun
python training.py \
    --model videomamba_middle \
    --data_path ${DATA_PATH} \
    --dataset 'celebdf' \
    --nb_classes 2 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --num_workers 8 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --delete_head \
    --semantic_loading True \
    --epochs 20 \
    --lr 1e-4 \
    --drop_path 0.8 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
    --bf16