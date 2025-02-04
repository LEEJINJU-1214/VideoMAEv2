from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """데이터클래스로 모델 설정 정보를 담는다.

    Attributes:
        model (str): 모델 아키텍처 이름 (예: 'vit_base_patch16_224').
        num_classes (int): 분류할 클래스 개수.
        num_frames (int): 영상 프레임 수.
        tubelet_size (int): 비디오 파편(tubelet) 크기.
        drop_path_rate (float): drop path 비율.
        checkpoint_path (str): 체크포인트 파일(.pth) 경로.
        input_size (int): 전처리 시 (H, W)를 이 값으로 맞춤.
        label_path (str): 클래스 레이블이 한 줄당 하나씩 적힌 텍스트 파일 경로.
    """

    model: str
    num_classes: int
    num_frames: int
    tubelet_size: int
    drop_path_rate: float
    checkpoint_path: str
    input_size: int
    label_path: str
