# src/search/indexer.py

"""
FAISS 기반 패션 이미지 인덱스 구축

역할:
- 전체 데이터셋 임베딩 → FAISS 인덱스 구축
- 인덱스 저장 / 로드
- 실시간 유사 이미지 검색

실무 포인트:
- IVF (Inverted File Index):
  클러스터 기반 근사 검색
  O(N) → O(√N) 속도 개선!
- 서버 시작 시 인덱스 로드 → 메모리 상주
  → 매 요청마다 디스크 읽기 없음!

자료구조:
- FAISS Index: 벡터 검색 전용 자료구조
  IVF: 클러스터 → 후보군 → 정밀 검색
- numpy array: 임베딩 백업 (재구축용)
- Dict (Hash Map): idx → 메타데이터 O(1)
- List: 검색 결과 순서 관리
"""

import pickle
import time
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FashionFAISSIndexer:
    """
    FAISS IVF 기반 패션 이미지 인덱서

    인덱스 구조:
    ┌─────────────────────────────────┐
    │  IVF (Inverted File Index)      │
    │                                 │
    │  전체 벡터                       │
    │  → K-means로 nlist개 클러스터   │
    │  → 검색 시 nprobe개만 탐색      │
    │  → O(N) → O(√N) 속도 개선!     │
    └─────────────────────────────────┘

    자료구조:
    ① FAISS IndexIVFFlat
       - 클러스터 기반 근사 최근접 이웃
       - Inner Product = 코사인 유사도
    ② Dict (Hash Map)
       - faiss_idx → 메타데이터 O(1) 조회
    ③ numpy array
       - 전체 임베딩 백업
       - 인덱스 재구축 시 사용
    """

    def __init__(
        self,
        embed_dim: int = 512,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """
        Args:
            embed_dim: 임베딩 차원 (CLIP: 512)
            nlist:     IVF 클러스터 수
                       경험칙: √N
                       10000장 → 100
                       44000장 → 200
            nprobe:    검색 시 탐색할 클러스터 수
                       nprobe↑ → 정확도↑, 속도↓
                       보통 nlist의 5~10%
        """
        self.embed_dim = embed_dim
        self.nlist     = nlist
        self.nprobe    = nprobe

        # FAISS 인덱스 초기화
        # Inner Product → L2 정규화 후 = 코사인 유사도
        quantizer = faiss.IndexFlatIP(embed_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            embed_dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.index.nprobe = nprobe

        # 자료구조: Dict (Hash Map)
        # faiss 내부 idx → 원본 메타데이터
        # O(1) 조회!
        self.idx_to_meta = {}

        # 임베딩 백업 (재구축용)
        self.embeddings_backup: Optional[np.ndarray] = None

        self.is_trained = False
        self.total      = 0

        logger.info(
            f"[FAISSIndexer] 초기화 완료 | "
            f"embed_dim={embed_dim} | "
            f"nlist={nlist} | nprobe={nprobe}"
        )

    def build(
        self,
        embeddings: np.ndarray,
        metadata_list: list,
    ) -> None:
        """
        임베딩 + 메타데이터로 FAISS 인덱스 구축

        Args:
            embeddings:     (N, 512) float32 배열
            metadata_list:  N개 메타데이터 dict 리스트

        알고리즘:
        ① K-means 학습 (클러스터 중심 찾기)
        ② 각 벡터를 클러스터에 할당
        ③ 메타데이터 Hash Map 구축
        """
        assert len(embeddings) == len(metadata_list), \
            "임베딩 수와 메타데이터 수가 다릅니다!"

        # float32 강제 변환
        # FAISS는 float32만 지원!
        embeddings = np.ascontiguousarray(
            embeddings.astype(np.float32)
        )

        N = len(embeddings)
        logger.info(f"[FAISSIndexer] 인덱스 구축 시작: {N:,}개")

        # ----------------------------------------
        # ① K-means 학습
        # nlist개 클러스터 중심 찾기
        # ----------------------------------------
        logger.info(
            f"[FAISSIndexer] 클러스터 학습 중... "
            f"(nlist={self.nlist})"
        )
        start = time.time()
        self.index.train(embeddings)
        train_time = time.time() - start
        self.is_trained = True
        logger.info(
            f"[FAISSIndexer] 클러스터 학습 완료! "
            f"({train_time:.1f}s)"
        )

        # ----------------------------------------
        # ② 벡터 추가
        # ----------------------------------------
        self.index.add(embeddings)
        self.total = self.index.ntotal
        logger.info(
            f"[FAISSIndexer] 벡터 추가 완료: "
            f"{self.total:,}개"
        )

        # ----------------------------------------
        # ③ 메타데이터 Hash Map 구축
        # faiss idx(0,1,2...) → 메타데이터
        # O(1) 조회를 위해 dict 사용
        # ----------------------------------------
        self.idx_to_meta = {
            i: meta
            for i, meta in enumerate(metadata_list)
        }

        # 임베딩 백업
        self.embeddings_backup = embeddings

        logger.info(
            f"[FAISSIndexer] 인덱스 구축 완료! "
            f"총 {self.total:,}개"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list:
        """
        쿼리 임베딩으로 유사 아이템 검색

        Args:
            query_embedding: (512,) 쿼리 벡터
            top_k:           반환할 결과 수

        Returns:
            results: [
                {
                    "rank":     1,
                    "score":    0.95,
                    "image_id": 15970,
                    "metadata": {...}
                },
                ...
            ]

        알고리즘:
        ① nprobe개 클러스터 탐색
        ② 클러스터 내 정확한 유사도 계산
        ③ Top-K 반환

        자료구조:
        - FAISS 내부: Priority Queue로 Top-K 관리
          O(K log N)
        - Dict: idx → meta O(1) 조회
        - List: 결과 순서 관리
        """
        assert self.is_trained, \
            "인덱스가 구축되지 않았습니다! build() 먼저 실행!"

        # (1, 512) 형태로 변환
        query = np.ascontiguousarray(
            query_embedding.astype(np.float32)
            .reshape(1, -1)
        )

        # FAISS 검색
        # distances: (1, top_k) 유사도 점수
        # indices:   (1, top_k) 결과 인덱스
        distances, indices = self.index.search(
            query, top_k
        )

        # 자료구조: List (결과 순서 관리)
        results = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0], indices[0]), 1
        ):
            if idx == -1:
                # 결과 부족 시 FAISS가 -1 반환
                continue

            # Hash Map O(1) 조회
            meta = self.idx_to_meta.get(int(idx), {})

            results.append({
                "rank":     rank,
                "score":    round(float(dist), 4),
                "image_id": meta.get("id", idx),
                "metadata": meta,
            })

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> list:
        """
        배치 쿼리 검색
        여러 쿼리를 한 번에 검색!

        Args:
            query_embeddings: (B, 512) 쿼리 배열
            top_k:            반환할 결과 수

        Returns:
            batch_results: B개의 결과 리스트
        """
        queries = np.ascontiguousarray(
            query_embeddings.astype(np.float32)
        )

        distances, indices = self.index.search(
            queries, top_k
        )

        # B개의 결과 리스트
        batch_results = []
        for b in range(len(queries)):
            results = []
            for rank, (dist, idx) in enumerate(
                zip(distances[b], indices[b]), 1
            ):
                if idx == -1:
                    continue
                meta = self.idx_to_meta.get(int(idx), {})
                results.append({
                    "rank":     rank,
                    "score":    round(float(dist), 4),
                    "image_id": meta.get("id", idx),
                    "metadata": meta,
                })
            batch_results.append(results)

        return batch_results

    def save(self, save_dir: str) -> None:
        """
        인덱스 저장

        저장 파일:
        - fashion.index: FAISS 인덱스
        - metadata.pkl:  Hash Map (idx → meta)
        - embeddings.npy: 임베딩 백업
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 저장
        faiss.write_index(
            self.index,
            str(save_path / "fashion.index")
        )

        # 메타데이터 저장 (Hash Map → pickle)
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.idx_to_meta, f)

        # 임베딩 백업 저장
        if self.embeddings_backup is not None:
            np.save(
                save_path / "embeddings.npy",
                self.embeddings_backup
            )

        logger.info(
            f"[FAISSIndexer] 저장 완료: {save_dir} | "
            f"{self.total:,}개 벡터"
        )

    def load(self, save_dir: str) -> None:
        """
        인덱스 로드
        서버 시작 시 한 번만 실행!
        """
        save_path = Path(save_dir)

        # FAISS 인덱스 로드
        self.index = faiss.read_index(
            str(save_path / "fashion.index")
        )
        self.index.nprobe = self.nprobe
        self.is_trained   = True
        self.total        = self.index.ntotal

        # 메타데이터 로드 (Hash Map 복원)
        with open(save_path / "metadata.pkl", "rb") as f:
            self.idx_to_meta = pickle.load(f)

        # 임베딩 백업 로드 (있을 경우)
        emb_path = save_path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings_backup = np.load(
                str(emb_path)
            )

        logger.info(
            f"[FAISSIndexer] 로드 완료: {save_dir} | "
            f"{self.total:,}개 벡터"
        )

    def get_stats(self) -> dict:
        """인덱스 통계 반환"""
        return {
            "total":     self.total,
            "embed_dim": self.embed_dim,
            "nlist":     self.nlist,
            "nprobe":    self.nprobe,
            "is_trained": self.is_trained,
        }