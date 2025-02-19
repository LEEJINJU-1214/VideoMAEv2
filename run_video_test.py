import os
import math
import time
import argparse
from typing import Tuple
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import torch
import torch.nn as nn
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

    if os.path.isdir(args.video_path):
        video_list = [
            os.path.join(args.video_path, vd)
            for vd in os.listdir(args.video_path)
        ]
    else:
        video_list = [args.video_path]

    for video in video_list:
        cap = cv2.VideoCapture(video)
        frame_list = deque(maxlen=args.num_frames)
        frame_cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            args.output_dir + video.split("/")[-1], fourcc, 30, (1280, 720)
        )
        check_list = deque(maxlen=5)
        event = "normal"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            if not frame_cnt % 4:  # 7.5fps
                # continue
                data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(data)
                if len(frame_list) == frame_list.maxlen:
                    frames = model.pre_process(np.array(frame_list))
                    outputs = model(frames)

                    probabilities = model.post_process(outputs)
                    max_indices = torch.argmax(probabilities, dim=1)
                    max_values = probabilities.gather(
                        1, max_indices.unsqueeze(1)
                    ).squeeze(1)

                    if max_values > 0.7:
                        check_list.append(max_indices)
                        counter = Counter(check_list)
                        most_common, _ = counter.most_common(1)[0]
                        event = model.label_names[most_common]

            cv2.putText(
                frame,
                event,
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            minutes = math.floor(((frame_cnt / 30) % 3600) / 60)
            seconds = (frame_cnt / 30) % 60

            cv2.putText(
                frame,
                f"{minutes} : {int(seconds)}",
                (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
