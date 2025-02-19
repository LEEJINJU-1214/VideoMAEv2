#!/usr/bin/env bash
set -x

# .env 파일 로드
if [ -f .env ]; then
    source .env
else
    echo ".env 파일을 찾을 수 없습니다."
    exit 1
fi

python run_video_test.py \
        --model vit_base_patch16_224 \
        --video_path ${TEST_VIDEO_PATH} \
        --image_dir ${TEST_IMAGE_DIR} \
        --checkpoint_path ${TEST_WEIGHT_PATH} \
        --output_dir ${TEST_OUTPUT_DIR} \
        --drop_path_rate 0.3  \
        --num_classes 2 \
        --input_size 224 \
        --num_frames 16 \
        --tubelet_size 2 \
        --label_path ${TEST_LABEL_PATH} 