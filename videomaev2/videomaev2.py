from typing import List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from timm.models import create_model
from torchvision import transforms

from .utils import (  # utils.load_state_dict()에서 사용
    get_transform,
    load_state_dict,
)
from .config import ModelConfig
from .models import *  # 필요한 경우 주석 해제 후 사용


def load_checkpoint_model(checkpoint_path: str) -> OrderedDict:
    """체크포인트에서 모델 가중치를 로드하여 OrderedDict 형태로 반환한다.

    체크포인트 딕셔너리에서 'model' 또는 'module' 키를 우선 탐색하고,
    특정 prefix('_orig_mod.', 'backbone.', 'encoder.' 등)를 제거한 뒤 반환한다.

    Args:
        checkpoint_path (str): 체크포인트 파일(.pth) 경로.

    Returns:
        OrderedDict: prefix 정리 후 최종 모델 가중치 딕셔너리.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # 1) 모델 가중치 정보 찾기
    if "model" in ckpt:
        checkpoint_model = ckpt["model"]
    elif "module" in ckpt:
        checkpoint_model = ckpt["module"]
    else:
        # 위 키가 없는 경우, 전체 ckpt 자체를 모델 가중치로 사용
        checkpoint_model = ckpt

    # 2) prefix("_orig_mod.") 제거
    for old_key in list(checkpoint_model.keys()):
        if old_key.startswith("_orig_mod."):
            new_key = old_key[len("_orig_mod.") :]
            checkpoint_model[new_key] = checkpoint_model.pop(old_key)

    # 3) backbone., encoder. 등 prefix 제거
    new_dict = OrderedDict()
    for key, val in checkpoint_model.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone.") :]
            new_dict[new_key] = val
        elif key.startswith("encoder."):
            new_key = key[len("encoder.") :]
            new_dict[new_key] = val
        else:
            new_dict[key] = val

    return new_dict


class VideoMaeV2(nn.Module):
    """VideoMaeV2 추론을 위한 딥러닝 모델 클래스.

    Attributes:
        model (nn.Module): timm.create_model()로 생성된 내부 모델.
        transform (transforms.Compose): 입력 영상을 텐서화하고 리사이즈하는 전처리 파이프라인.
        label_names (List[str]): 레이블(클래스) 이름 목록.
        softmax (nn.Softmax): 모델 출력에 대한 소프트맥스 함수.
    """

    def __init__(self, model_config: ModelConfig):
        """VideoMaeV2 모델 초기화.

        timm 라이브러리로 모델을 생성하고, 필요하면 체크포인트를 로드한다.
        또한 전처리용 transform과 라벨 정보를 세팅한다.

        Args:
            model_config (ModelConfig): 모델 설정 정보를 담은 데이터클래스.
        """
        super().__init__()

        with open(model_config.label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.label_names: List[str] = [line.strip() for line in lines]


        self.model = create_model(
            model_config.model,
            img_size=model_config.input_size,
            pretrained=False,
            num_classes=len(self.label_names),
            all_frames=model_config.num_frames,
            tubelet_size=model_config.tubelet_size,
            drop_path_rate=model_config.drop_path_rate,
            use_mean_pooling=True,
        )

        if model_config.checkpoint_path:
            checkpoint_model = load_checkpoint_model(
                model_config.checkpoint_path
            )
            load_state_dict(self.model, checkpoint_model)

        self.transform = get_transform(
            (model_config.input_size, model_config.input_size)
        )

        self.softmax = nn.Softmax(dim=1)
        self.model.eval()

    def pre_process(self, raw_frames: np.ndarray) -> torch.Tensor:
        """영상(np.ndarray 형식)을 받아 전처리 후 GPU 텐서로 변환한다.

        (T, H, W, C) 형태의 NumPy 배열을 (B, C, T, H, W) 텐서로 변환하며,
        값 범위도 [0, 1]로 조정된다.

        Args:
            raw_frames (np.ndarray): 원본 영상. (T, H, W, C) 형태.

        Returns:
            torch.Tensor: (1, C, T, H, W) 형태의 전처리된 GPU 텐서.
        """
        vid_tensor = torch.tensor(raw_frames)  # (T, H, W, C)
        transformed = self.transform(vid_tensor)  # (C, T, H, W)
        return transformed.unsqueeze(0).cuda()  # (B=1, C, T, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """추론을 수행한다.

        Args:
            x (torch.Tensor): (B, C, T, H, W) 형태의 입력 텐서.

        Returns:
            torch.Tensor: 모델 출력 로짓(분류 결과 전 단계).
        """
        return self.model(x)

    def post_process(self, outputs: torch.Tensor) -> torch.Tensor:
        """소프트맥스를 적용하여 확률 벡터를 반환한다.

        Args:
            outputs (torch.Tensor): 모델의 로짓 출력.

        Returns:
            torch.Tensor: 소프트맥스가 적용된 확률 벡터.
        """
        return self.softmax(outputs)
