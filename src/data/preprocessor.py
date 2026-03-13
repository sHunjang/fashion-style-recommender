# src/data/preprocessor.py

"""
Fashion Image Preprocessor

역할:
- CLIP 입력용 이미지 전처리
- 학습 / 추론 전처리 파이프라인 분리
- 데이터 증강 (Augmentation)
- 배치 전처리

실무 포인트:
- 학습 시: 증강 적용 (과적합 방지)
- 추론 시: 증강 없이 일관된 전처리만!
  → 같은 이미지는 항상 같은 임베딩!

자료구조:
- List: 전처리 파이프라인 순서 관리
- Dict: 전처리 설정 관리
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


# ----------------------------------------
# CLIP 공식 정규화 값
# ImageNet 값과 다르므로 주의!
# ----------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class FashionPreprocessor:
    """
    Fashion 이미지 전처리 파이프라인

    학습용 (train):
    RandomCrop + HorizontalFlip + ColorJitter
    → 다양한 변형으로 일반화 성능 향상!

    추론용 (inference):
    Resize + CenterCrop만 적용
    → 일관된 전처리로 안정적인 임베딩!

    자료구조:
    - transforms.Compose: 파이프라인을 List로 관리
      순서가 중요! (Resize → Crop → Normalize)
    """

    def __init__(
        self,
        image_size: int = 224,
        mode: str = "inference",
    ):
        """
        Args:
            image_size: 출력 이미지 크기 (CLIP: 224)
            mode:       'train' | 'inference'
        """
        self.image_size = image_size
        self.mode       = mode

        # 파이프라인 구성
        self.train_transform     = self._build_train()
        self.inference_transform = self._build_inference()

        print(f"[FashionPreprocessor] mode={mode} | "
              f"image_size={image_size}")

    def _build_train(self) -> transforms.Compose:
        """
        학습용 전처리 파이프라인
        데이터 증강 포함!

        파이프라인 순서 (자료구조: List):
        ① Resize         : 256으로 리사이즈
        ② RandomCrop     : 224로 랜덤 크롭
        ③ HorizontalFlip : 50% 확률로 좌우 반전
        ④ ColorJitter    : 색상 랜덤 변형
        ⑤ ToTensor       : PIL → Tensor
        ⑥ Normalize      : CLIP 정규화
        """
        return transforms.Compose([
            # ① 약간 크게 리사이즈
            transforms.Resize(
                256,
                interpolation=\
                    transforms.InterpolationMode.BICUBIC
            ),

            # ② 랜덤 크롭
            # 이미지의 다양한 부분 학습!
            transforms.RandomCrop(self.image_size),

            # ③ 좌우 반전
            # 패션: 좌우 대칭이 많아서 효과적!
            transforms.RandomHorizontalFlip(p=0.5),

            # ④ 색상 증강
            # 같은 옷이라도 조명에 따라 색이 다름!
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                # hue는 작게! 색상이 너무 바뀌면 안 됨
            ),

            # ⑤ Tensor 변환
            transforms.ToTensor(),

            # ⑥ CLIP 정규화
            transforms.Normalize(
                mean=CLIP_MEAN,
                std=CLIP_STD
            ),
        ])

    def _build_inference(self) -> transforms.Compose:
        """
        추론용 전처리 파이프라인
        증강 없음! 일관성이 핵심!

        파이프라인 순서 (자료구조: List):
        ① Resize      : 224로 리사이즈
        ② CenterCrop  : 중앙 크롭
        ③ ToTensor    : PIL → Tensor
        ④ Normalize   : CLIP 정규화
        """
        return transforms.Compose([
            # ① 리사이즈
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=\
                    transforms.InterpolationMode.BICUBIC
            ),

            # ② 중앙 크롭
            # 항상 같은 영역 → 일관된 임베딩!
            transforms.CenterCrop(self.image_size),

            # ③ Tensor 변환
            transforms.ToTensor(),

            # ④ CLIP 정규화
            transforms.Normalize(
                mean=CLIP_MEAN,
                std=CLIP_STD
            ),
        ])

    def __call__(
        self,
        image: Union[Image.Image, str, Path],
    ) -> torch.Tensor:
        """
        이미지 전처리 실행

        Args:
            image: PIL Image | 이미지 파일 경로

        Returns:
            tensor: (3, 224, 224) 전처리된 텐서
        """
        # 경로 입력 시 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # PIL이 아니면 변환
        if not isinstance(image, Image.Image):
            raise ValueError(
                f"지원하지 않는 입력 타입: {type(image)}"
            )

        # RGB 강제 변환
        # RGBA, 흑백 등 다양한 포맷 대응
        image = image.convert("RGB")

        # 모드에 따라 파이프라인 선택
        if self.mode == "train":
            return self.train_transform(image)
        else:
            return self.inference_transform(image)

    def preprocess_batch(
        self,
        images: list,
    ) -> torch.Tensor:
        """
        이미지 배치 전처리

        Args:
            images: PIL Image 또는 경로 리스트

        Returns:
            tensor: (N, 3, 224, 224)

        자료구조:
        - List → 순서대로 전처리
        - torch.stack → 배치 텐서로 합치기
        """
        tensors = [self(img) for img in images]
        return torch.stack(tensors)
        # (N, 3, 224, 224)

    def decode_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Image.Image:
        """
        텐서 → PIL Image 역변환
        시각화 및 디버깅용

        Args:
            tensor: (3, 224, 224)

        Returns:
            PIL Image
        """
        # CLIP 정규화 역변환
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(3, 1, 1)

        tensor = tensor.cpu().clone()
        tensor = tensor * std + mean
        # 역정규화: x = (x_norm * std) + mean

        # 클리핑 (0~1 범위 유지)
        tensor = tensor.clamp(0, 1)

        # numpy 변환
        np_image = (
            tensor.permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)

        return Image.fromarray(np_image)

    def set_mode(self, mode: str) -> None:
        """
        전처리 모드 전환

        Args:
            mode: 'train' | 'inference'
        """
        assert mode in ("train", "inference"), \
            "mode는 'train' 또는 'inference'!"
        self.mode = mode
        print(f"[FashionPreprocessor] mode → {mode}")