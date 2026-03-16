# src/models/clip_encoder.py

"""
CLIP 기반 패션 이미지/텍스트 인코더

역할:
- 이미지  → 512d 임베딩 벡터
- 텍스트  → 512d 임베딩 벡터
- 같은 스타일 = 벡터 공간에서 가까운 위치!

실무 포인트:
- 모델은 서버 시작 시 한 번만 로드!
- 배치 처리로 GPU 효율 최대화
- @torch.no_grad() 로 추론 메모리 절약
- float16 (half precision) 으로 속도 향상

자료구조:
- numpy array: 임베딩 벡터 저장
  → FAISS 입력 형식
  → 연속 메모리 → 빠른 행렬 연산
- List: 배치 단위 결과 누적
"""

import math
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FashionCLIPEncoder:
    """
    CLIP 기반 패션 임베딩 인코더

    이미지와 텍스트를 동일한 512차원
    임베딩 공간으로 변환

    핵심 원리:
    같은 스타일의 이미지와 텍스트
    → 코사인 유사도 높음 (벡터 방향 유사)
    → FAISS 검색으로 빠르게 찾기!
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        use_fp16: bool = True,
        lora_path: str = None,
    ):
        """
        Args:
            model_name: HuggingFace CLIP 모델명
            device:     'cuda' | 'cpu'
            use_fp16:   float16 사용 여부
                        GPU 메모리 절약 + 속도 향상!
            lora_path:  LoRA 어댑터 경로
                        None → Zero-shot (기존)
                        경로 → Fine-tuned 모델 로드
        """
        # 디바이스 설정
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            and device == "cuda" else "cpu"
        )

        # float16: GPU에서 속도 2배 향상!
        # float32: CPU에서는 float16 미지원
        self.use_fp16 = (
            use_fp16
            and self.device.type == "cuda"
        )
        self.dtype = (
            torch.float16
            if self.use_fp16
            else torch.float32
        )

        logger.info(
            f"[CLIPEncoder] 디바이스: {self.device} | "
            f"dtype: {self.dtype}"
        )

        # ----------------------------------------
        # CLIP 모델 로드
        # HuggingFace Hub에서 자동 다운로드
        # 최초 1회만 다운로드 → 캐시에 저장
        # ----------------------------------------
        logger.info(f"[CLIPEncoder] 모델 로딩: {model_name}")

        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

        # 추론 모드 고정
        # 드롭아웃 비활성화 + 배치 정규화 고정
        self.model.eval()
        
        # ----------------------------------------
        # LoRA 어댑터 로드 (파인튜닝 모델)
        # ----------------------------------------
        self._is_finetuned= False
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            logger.info(
                f"[CLIPEncoder] LoRA 로드: {lora_path}"
            )
            self.model = PeftModel.from_pretrained(
                self.model, lora_path
            )
            # LoRA 가중치를 기존 가중치에 병합!
            # → 추론 속도 Zero-shot과 동일하게 유지!
            self.model = self.model.merge_and_unload()
            self.model.eval()
            self._is_finetuned = True
            logger.info(
                "[CLIPEncoder] LoRA 병합 완료! "
                "(Fine-tuned 모드)"
            )
        else:
            logger.info(
                "[CLIPEncoder] Zero-shot 모드"
            )


        # 임베딩 차원 확인
        self.embed_dim = (
            self.model.config.projection_dim
        )

        logger.info(
            f"[CLIPEncoder] 로드 완료! "
            f"embed_dim={self.embed_dim}"
        )

        # GPU 메모리 정보 출력
        if self.device.type == "cuda":
            mem_allocated = torch.cuda.memory_allocated() \
                            / 1024**2
            logger.info(
                f"[CLIPEncoder] "
                f"GPU 메모리 사용: {mem_allocated:.1f}MB"
            )

    @torch.no_grad()
    def encode_images(
        self,
        images: list,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        이미지 리스트 → 임베딩 배열

        실무 포인트:
        배치 처리로 GPU 효율 최대화!
        한 번에 너무 많이 넣으면 OOM 발생
        → batch_size로 조절!

        Args:
            images:        PIL Image 리스트
            batch_size:    배치 크기 (GPU 메모리에 따라 조절)
            show_progress: tqdm 진행바 표시

        Returns:
            embeddings: (N, 512) float32 numpy array
                        L2 정규화 완료!

        자료구조:
        - List: 배치별 결과 누적
        - numpy vstack: 최종 (N, 512) 배열 합치기
        """
        # 자료구조: List (배치 결과 누적)
        all_embeddings = []

        # tqdm: 진행 상황 시각화
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Image Encoding",
                total=math.ceil(len(images) / batch_size)
            )

        for i in iterator:
            batch = images[i: i + batch_size]

            # CLIP 전처리
            # Resize(224) + Normalize(CLIP 기준)
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            )

            # dtype 맞춰서 GPU로 이동
            inputs = {
                k: v.to(self.device)
                if v.dtype == torch.float32
                else v.to(self.device)
                for k, v in inputs.items()
            }

            # ----------------------------------------
            # Transformers 5.x 버전 대응
            # get_image_features() → 텐서 직접 반환
            # ----------------------------------------
            image_features = \
                self.model.get_image_features(**inputs)

            # L2 정규화
            # 코사인 유사도 = 단순 내적!
            # similarity(a,b) = a·b (정규화 후)
            # Transformers 5.x: 객체 또는 텐서 모두 대응
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output

            # L2 정규화
            image_features = F.normalize(
                image_features.float(), dim=-1
            )

            all_embeddings.append(
                image_features.cpu().numpy()
            )


        # 자료구조: numpy vstack
        # List[(64,512), (64,512)...] → (N, 512)
        return np.vstack(all_embeddings).astype(
            np.float32
        )

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list,
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        텍스트 리스트 → 임베딩 배열

        텍스트는 이미지보다 훨씬 빠름!
        → batch_size 크게 설정 가능

        Args:
            texts:         텍스트 리스트
            batch_size:    배치 크기
            show_progress: 진행바 표시

        Returns:
            embeddings: (N, 512) float32 numpy array
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Text Encoding",
                total=math.ceil(len(texts) / batch_size)
            )

        for i in iterator:
            batch = texts[i: i + batch_size]

            # 텍스트 토크나이징
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
                # CLIP 최대 토큰 길이!
            ).to(self.device)

            # ----------------------------------------
            # Transformers 5.x 버전 대응
            # ----------------------------------------
            text_features = \
                self.model.get_text_features(**inputs)

            # 객체로 반환될 경우 텐서 추출
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output

            # L2 정규화
            text_features = F.normalize(
                text_features.float(), dim=-1
            )

            all_embeddings.append(
                text_features.cpu().numpy()
            )

        return np.vstack(all_embeddings).astype(np.float32)

    @torch.no_grad()
    def encode_single_image(
        self,
        image: Union[Image.Image, str, Path],
    ) -> np.ndarray:
        """
        단일 이미지 인코딩
        실시간 검색 쿼리용!

        Args:
            image: PIL Image 또는 파일 경로

        Returns:
            embedding: (512,) float32 numpy array
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        return self.encode_images(
            [image],
            batch_size=1,
            show_progress=False,
        )[0]

    @torch.no_grad()
    def encode_single_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        단일 텍스트 인코딩
        실시간 텍스트 검색용!

        Args:
            text: 검색 텍스트
                  예: "casual blue denim jacket"

        Returns:
            embedding: (512,) float32 numpy array
        """
        return self.encode_texts(
            [text],
            show_progress=False,
        )[0]

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        두 임베딩 간 코사인 유사도 계산

        L2 정규화된 벡터끼리는
        내적 = 코사인 유사도!

        Args:
            embedding1: (512,) 벡터
            embedding2: (512,) 벡터

        Returns:
            similarity: -1.0 ~ 1.0
                        1.0: 완전히 같은 스타일
                        0.0: 관련 없음
                       -1.0: 완전히 반대
        """
        # L2 정규화 확인 후 내적
        e1 = embedding1 / (
            np.linalg.norm(embedding1) + 1e-8
        )
        e2 = embedding2 / (
            np.linalg.norm(embedding2) + 1e-8
        )
        return float(np.dot(e1, e2))

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        info = {
            "model":     "CLIP ViT-B/32",
            "embed_dim": self.embed_dim,
            "device":    str(self.device),
            "dtype":     str(self.dtype),
            "use_fp16":  self.use_fp16,
            "mode": "fine-tuned" if self._is_finetuned else "zero-shot",
        }

        if self.device.type == "cuda":
            info["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )
            info["gpu_memory_mb"] = round(
                torch.cuda.memory_allocated() / 1024**2,
                1
            )

        return info