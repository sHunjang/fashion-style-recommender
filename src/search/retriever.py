# src/search/retriever.py

"""
Fashion Style Retriever

역할:
- 이미지/텍스트 쿼리 → 유사 스타일 검색
- 전체 파이프라인 통합:
  입력 → CLIP 인코딩 → FAISS 검색 → 결과 반환

실무 포인트:
- Retriever = 인코더 + 인덱서 통합 인터페이스
- API 서버에서 이 클래스만 사용!
- 인덱스 구축 ~ 검색까지 원스톱!

자료구조:
- Dict: 검색 결과 포맷
- List: Top-K 결과 순서 관리
"""

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer

logger = logging.getLogger(__name__)


class FashionRetriever:
    """
    패션 스타일 검색 엔진

    전체 파이프라인:
    이미지/텍스트 입력
        ↓
    CLIP 인코딩 (512d 임베딩)
        ↓
    FAISS IVF 검색 (Top-K)
        ↓
    메타데이터 조회 (Hash Map)
        ↓
    결과 반환
    """

    def __init__(
        self,
        encoder: FashionCLIPEncoder,
        indexer: FashionFAISSIndexer,
    ):
        """
        Args:
            encoder: FashionCLIPEncoder 인스턴스
            indexer: FashionFAISSIndexer 인스턴스
        """
        self.encoder = encoder
        self.indexer = indexer
        logger.info("[Retriever] 초기화 완료!")

    def build_index_from_dataset(
        self,
        dataset: FashionDataset,
        batch_size: int = 64,
    ) -> None:
        """
        데이터셋 전체로 FAISS 인덱스 구축

        Args:
            dataset:    FashionDataset 인스턴스
            batch_size: 인코딩 배치 크기
        """
        logger.info(
            f"[Retriever] 인덱스 구축 시작: "
            f"{len(dataset):,}개"
        )

        # ① 전체 이미지 로드
        logger.info("[Retriever] 이미지 로드 중...")
        all_metadata = dataset.get_all_metadata()

        images = []
        valid_metadata = []

        for meta in all_metadata:
            try:
                img = Image.open(
                    meta["image_path"]
                ).convert("RGB")
                images.append(img)
                valid_metadata.append(meta)
            except Exception as e:
                logger.warning(
                    f"이미지 로드 실패: "
                    f"{meta['image_path']} | {e}"
                )

        logger.info(
            f"[Retriever] 이미지 로드 완료: "
            f"{len(images):,}개"
        )

        # ② CLIP 임베딩 추출
        logger.info("[Retriever] CLIP 임베딩 추출 중...")
        start = time.time()
        embeddings = self.encoder.encode_images(
            images,
            batch_size=batch_size,
            show_progress=True,
        )
        encode_time = time.time() - start
        logger.info(
            f"[Retriever] 임베딩 완료: "
            f"{encode_time:.1f}s | "
            f"shape={embeddings.shape}"
        )

        # ③ FAISS 인덱스 구축
        self.indexer.build(embeddings, valid_metadata)
        logger.info("[Retriever] 인덱스 구축 완료!")

    def search_by_image(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = 10,
    ) -> dict:
        """
        이미지로 유사 스타일 검색

        Args:
            image: PIL Image 또는 파일 경로
            top_k: 반환할 결과 수

        Returns:
            {
                "query_type": "image",
                "search_time_ms": 12.3,
                "results": [...]
            }
        """
        start = time.time()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # CLIP 인코딩
        query_emb = self.encoder.encode_single_image(image)

        # FAISS 검색
        results = self.indexer.search(query_emb, top_k)

        elapsed_ms = (time.time() - start) * 1000

        return {
            "query_type":     "image",
            "search_time_ms": round(elapsed_ms, 1),
            "total_results":  len(results),
            "results":        results,
        }

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
    ) -> dict:
        """
        텍스트로 유사 스타일 검색
        Zero-shot! 학습 없이 텍스트로 검색!

        Args:
            text:  스타일 설명
                   예: "casual blue denim jacket"
            top_k: 반환할 결과 수

        Returns:
            {
                "query_type": "text",
                "query": "casual blue denim jacket",
                "search_time_ms": 8.5,
                "results": [...]
            }
        """
        start = time.time()

        # CLIP 텍스트 인코딩
        query_emb = self.encoder.encode_single_text(text)

        # FAISS 검색
        results = self.indexer.search(query_emb, top_k)

        elapsed_ms = (time.time() - start) * 1000

        return {
            "query_type":     "text",
            "query":          text,
            "search_time_ms": round(elapsed_ms, 1),
            "total_results":  len(results),
            "results":        results,
        }

    def save_index(self, save_dir: str) -> None:
        """인덱스 저장"""
        self.indexer.save(save_dir)
        logger.info(f"[Retriever] 인덱스 저장: {save_dir}")

    def load_index(self, save_dir: str) -> None:
        """인덱스 로드"""
        self.indexer.load(save_dir)
        logger.info(f"[Retriever] 인덱스 로드: {save_dir}")

    def get_stats(self) -> dict:
        """검색 엔진 통계"""
        return {
            "encoder": self.encoder.get_model_info(),
            "index":   self.indexer.get_stats(),
        }