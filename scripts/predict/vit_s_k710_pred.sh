#!/usr/bin/env bash
set -x

OUTPUT_DIR='logs/'
VIDEO_PATH= '0'
CKPT_PATH='weights/vit_s_k710_dl_from_giant.pth'
LABEL_PATH='misc/label_map_k710.txt'

python -m predict.py \
        --model vit_base_patch16_224 \
        --video_path ${DATA_PATH} \
        --checkpoint_path ${CKPT_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --drop_path_rate 0.3  \
        --num_classes 710 \
        --input_size 224 \
        --num_frames 16 \
        --tubelet_size 2 \
        --label_path ${LABEL_PATH} 