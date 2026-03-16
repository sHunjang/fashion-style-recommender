# src/detection/pipeline.py

"""
패션 탐지 + CLIP 검색 통합 파이프라인

전체 흐름:
코디 이미지 1장
    ↓
YOLOv8 탐지 (아이템 위치 찾기)
    ↓
각 아이템 크롭
    ↓
CLIP LoRA 임베딩 (512d 벡터)
    ↓
FAISS 검색 (유사 상품 찾기)
    ↓
아이템별 Top-K 추천 결과 반환

실무 가치:
"인스타 코디 사진 1장으로
 전체 아이템 쇼핑 가능!"
"""

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src.detection.detector import FashionDetector
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

logger = logging.getLogger(__name__)


class FashionPipeline:
    """
    패션 탐지 + 검색 통합 파이프라인

    구성 요소:
    ① FashionDetector  → YOLOv8 탐지
    ② FashionRetriever → CLIP + FAISS 검색

    사용법:
    pipeline = FashionPipeline(...)
    results  = pipeline.search(image)
    → 아이템별 유사 상품 리스트 반환!
    """

    def __init__(
        self,
        detector:  FashionDetector,
        retriever: FashionRetriever,
        top_k:     int   = 5,
        min_confidence: float = 0.25,
    ):
        """
        Args:
            detector:       FashionDetector 인스턴스
            retriever:      FashionRetriever 인스턴스
            top_k:          아이템별 추천 수
            min_confidence: 최소 탐지 신뢰도
        """
        self.detector       = detector
        self.retriever      = retriever
        self.top_k          = top_k
        self.min_confidence = min_confidence

        logger.info(
            f"[Pipeline] 초기화 완료! | "
            f"top_k={top_k} | "
            f"min_confidence={min_confidence}"
        )

    def search(
        self,
        image: Union[Image.Image, str, Path],
        only_searchable: bool = True,
    ) -> dict:
        """
        코디 이미지 → 아이템별 유사 상품 검색

        Args:
            image:           PIL Image 또는 경로
            only_searchable: True → 검색 가능 클래스만

        Returns:
            {
                "total_items":   3,
                "search_time_ms": 45.2,
                "items": [
                    {
                        "rank":        1,
                        "class_name":  "Shirt",
                        "confidence":  0.95,
                        "bbox":        [x1,y1,x2,y2],
                        "crop":        PIL Image,
                        "similar":     [
                            {
                                "rank":     1,
                                "score":    0.85,
                                "image_id": 15970,
                                "metadata": {...}
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        start = time.time()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # ----------------------------------------
        # STEP 1: YOLOv8 탐지
        # ----------------------------------------
        detections = self.detector.detect(
            image,
            only_searchable=only_searchable,
        )

        if not detections:
            logger.info("[Pipeline] 탐지된 아이템 없음!")
            return {
                "total_items":    0,
                "search_time_ms": round(
                    (time.time() - start) * 1000, 1
                ),
                "items": [],
            }

        logger.info(
            f"[Pipeline] 탐지 완료: "
            f"{len(detections)}개 아이템"
        )

        # ----------------------------------------
        # STEP 2: 각 아이템 크롭 → CLIP 검색
        # ----------------------------------------
        items = []
        for det in detections:
            crop       = det["crop"]
            class_name = det["class_name"]

            # CLIP 임베딩 + FAISS 검색
            similar = self.retriever.search_by_image(
                crop,
                top_k=self.top_k,
            )

            items.append({
                "rank":       det["rank"],
                "class_name": class_name,
                "confidence": det["confidence"],
                "bbox":       det["bbox"],
                "crop":       crop,
                "similar":    similar["results"],
                "search_time_ms": similar["search_time_ms"],
            })

            logger.info(
                f"  [{class_name}] "
                f"유사 상품 {len(similar['results'])}개 검색 완료"
            )

        elapsed_ms = round(
            (time.time() - start) * 1000, 1
        )

        logger.info(
            f"[Pipeline] 전체 완료: "
            f"{elapsed_ms}ms"
        )

        return {
            "total_items":    len(items),
            "search_time_ms": elapsed_ms,
            "items":          items,
        }

    def search_and_visualize(
        self,
        image: Union[Image.Image, str, Path],
        only_searchable: bool = True,
    ) -> tuple:
        """
        탐지 + 검색 + 시각화 한 번에!

        Returns:
            (results, visualized_image)
            visualized_image: 탐지 박스 그려진 이미지
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        results = self.search(image, only_searchable)

        # 탐지 결과로 시각화
        detections = [
            {
                "rank":       item["rank"],
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "bbox":       item["bbox"],
                "crop":       item["crop"],
            }
            for item in results["items"]
        ]

        visualized = self.detector.visualize(
            image, detections
        )

        return results, visualized

    def get_stats(self) -> dict:
        """파이프라인 통계"""
        return {
            "detector":  self.detector.get_model_info(),
            "retriever": self.retriever.get_stats(),
            "top_k":     self.top_k,
        }