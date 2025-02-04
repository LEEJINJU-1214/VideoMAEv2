import time
import argparse
from typing import Tuple
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from pyparsing import deque
from timm.models import create_model
from torchvision import transforms

from videomaev2 import VideoMaeV2, ModelConfig


def get_args():
    """
    Get VideoMAE v2 predict script arguments.

    Returns:
        args: A argparse.Namespace object containing the following attributes:
            model: The name of the model to predict.
            video_path: The path to the video file.
            image_dir: The directory containing the images.
            checkpoint_path: The path to the checkpoint file.
            output_dir: The directory to save the output.
            drop_path_rate: The dropout path rate.
            num_classes: The number of classes.
            input_size: The size of the input.
            num_frames: The number of frames.
            tubelet_size: The size of the tubelet.
            label_path: The path to the label file.
            use_cuda: Whether to use CUDA or not.
    """
    parser = argparse.ArgumentParser(
        "VideoMAE v2 predict script", add_help=False
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to predict",
    )

    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--drop_path_rate", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=710)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--use_cuda", action="store_true", default=True)

    return parser.parse_args()


def main(args):
    model = VideoMaeV2(
        ModelConfig(
            model=args.model,
            checkpoint_path=args.checkpoint_path,
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            drop_path_rate=args.drop_path_rate,
            input_size=args.input_size,
            label_path=args.label_path,
        )
    )
    model.eval()
    if args.use_cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    cap = cv2.VideoCapture(args.video_path)
    frame_list = deque(maxlen=args.num_frames)
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1
        if frame_cnt % 4:  # 7.5fps
            continue
        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(data)
        if len(frame_list) != frame_list.maxlen:
            continue
        frames = model.pre_process(np.array(frame_list))
        outputs = model(frames)
        probabilities = model.post_process(outputs)
        for i in range(7):
            frame_list.popleft()

        # 특정 값 이상일 때의 확률과 인덱스 구하기
        threshold = 0.5  # 설정한 임계값
        mask = probabilities >= threshold
        filtered_probabilities = probabilities[mask]  # 임계값 이상인 확률
        filtered_indices = torch.nonzero(mask.flatten())

        for score, index in zip(filtered_probabilities, filtered_indices):
            print(model.label_names[index], score, index)
            cv2.imwrite(
                args.output_dir
                + str(frame_cnt)
                + "_"
                + model.label_names[index]
                + "_score:"
                + str(round(score.item(), 2))
                + ".jpg",
                frame,
            )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
