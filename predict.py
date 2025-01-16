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

import utils
import models


def get_args():
    parser = argparse.ArgumentParser(
        "VideoMAE v2 predict script", add_help=False
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_small_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to predict",
    )

    parser.add_argument("--video_path", type=str, default="0")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--drop_path_rate", type=int, default=0.3)
    parser.add_argument("--num_classes", type=int, default=710)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--label_path", type=str)

    parser.add_argument("--use_cuda", action="store_true", default=True)

    return parser.parse_args()


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation="bilinear"):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False,
    )


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_transform(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([ToFloatTensorInZeroOne(), Resize(size)])


def get_model(args):
    # get model & load ckpt
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        img_size=args.img_size,
        pretrained=False,
        num_classes=args.num_classes,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path_rate=args.drop_path_rate,
        use_mean_pooling=True,
    )
    if args.checkpoint_path:
        load_weights(model, args)
    return model


def load_weights(model, args):
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    checkpoint_model = None
    model_key = "model|module".split("|")
    for model_key in ["model", "module"]:
        if model_key in ckpt:
            checkpoint_model = ckpt[model_key]
            break
    if checkpoint_model is None:
        checkpoint_model = ckpt
    for old_key in list(checkpoint_model.keys()):
        if old_key.startswith("_orig_mod."):
            new_key = old_key[10:]
            checkpoint_model[new_key] = checkpoint_model.pop(old_key)

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith("backbone."):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith("encoder."):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    utils.load_state_dict(model, checkpoint_model)


# ckpt = torch.load("weights/vit_s_k710_dl_from_giant.pth", map_location="cpu")

# checkpoint_model = None
# model_key = "model|module".split("|")
# for model_key in ["model", "module"]:
#     if model_key in ckpt:
#         checkpoint_model = ckpt[model_key]
#         break
# if checkpoint_model is None:
#     checkpoint_model = ckpt
# for old_key in list(checkpoint_model.keys()):
#     if old_key.startswith("_orig_mod."):
#         new_key = old_key[10:]
#         checkpoint_model[new_key] = checkpoint_model.pop(old_key)

# all_keys = list(checkpoint_model.keys())
# new_dict = OrderedDict()
# for key in all_keys:
#     if key.startswith("backbone."):
#         new_dict[key[9:]] = checkpoint_model[key]
#     elif key.startswith("encoder."):
#         new_dict[key[8:]] = checkpoint_model[key]
#     else:
#         new_dict[key] = checkpoint_model[key]
# checkpoint_model = new_dict

# utils.load_state_dict(model, checkpoint_model)

# model.eval()
# model.to("cuda")
# transform = get_transform((224, 224))

# softmax = nn.Softmax(dim=1)  # dim=1은 클래스 차원에 대해 소프트맥스를 적용

# with open("misc/label_map_k710.txt", "r", encoding="utf-8") as file:
#     lines = file.readlines()

# label_names = [line.strip() for line in lines]
# # print(label_names)
# cap = cv2.VideoCapture(0)
# frame_list = deque(maxlen=16)
# start_t = time.time()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     if time.time() - start_t < 0.05:  ## 0.05초가 간격으로 프레임쌓음
#         continue
#     start_t = time.time()
#     frame_h, frame_w = frame.shape[:2]
#     frame = frame[100 : frame_h // 2, frame_w // 2 + 50 : frame_w - 200]
#     data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_list.append(data)
#     if len(frame_list) == 16:
#         final_data = torch.tensor(np.array(frame_list))
#         result = transform(final_data).unsqueeze(0).cuda()
#         outputs = model(result)
#         probabilities = softmax(outputs)
#         # 특정 값 이상일 때의 확률과 인덱스 구하기

#         threshold = 0.3  # 설정한 임계값
#         mask = probabilities >= threshold

#         filtered_probabilities = probabilities[mask]  # 임계값 이상인 확률
#         filtered_indices = torch.nonzero(mask.flatten())

#         for score, index in zip(filtered_probabilities, filtered_indices):
#             print(label_names[index], score, index)
#             cv2.imwrite(
#                 label_names[index]
#                 + "_"
#                 + str(round(score.item(), 2))
#                 + "_test.jpg",
#                 frame,
#             )

#     cv2.imwrite("frame.jpg", frame)

#     # cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()


def main(args):
    print(args)
    model = get_model(args)
    model.eval()
    if args.use_cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    transform = get_transform((args.input_size, args.input_size))
    softmax = nn.Softmax(dim=1)
    with open(args.label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    label_names = [line.strip() for line in lines]

    cap = cv2.VideoCapture(args.video_path)
    frame_list = deque(maxlen=args.num_frames)
    start_t = time.time()
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1
        if time.time() - start_t < 0.05:  ## 0.05초가 간격으로 프레임쌓음
            continue
        start_t = time.time()
        # frame_h, frame_w = frame.shape[:2]
        # frame = frame[100 : frame_h // 2, frame_w // 2 + 50 : frame_w - 200]
        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(data)
        if len(frame_list) != frame_list.maxlen:
            continue
        final_data = torch.tensor(np.array(frame_list))
        result = transform(final_data).unsqueeze(0).cuda()
        outputs = model(result)

        probabilities = softmax(outputs)
        # 특정 값 이상일 때의 확률과 인덱스 구하기
        threshold = 0.3  # 설정한 임계값
        mask = probabilities >= threshold
        filtered_probabilities = probabilities[mask]  # 임계값 이상인 확률
        filtered_indices = torch.nonzero(mask.flatten())

        for score, index in zip(filtered_probabilities, filtered_indices):
            print(label_names[index], score, index)
            cv2.imwrite(
                args.output_dir
                + label_names[index]
                + "_frame:"
                + str(frame_cnt)
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
