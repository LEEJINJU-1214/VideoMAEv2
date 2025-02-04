#!/usr/bin/env bash
set -x  # print the commands


# .env 파일 로드
if [ -f .env ]; then
    source .env
else
    echo ".env 파일을 찾을 수 없습니다."
    exit 1
fi

SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 torchrun --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set Custom_Image \
        --nb_classes 2 \
        --data_path ${TRAIN_CSV_DIR} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 3 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 35 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
