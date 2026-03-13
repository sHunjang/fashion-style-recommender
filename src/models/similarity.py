# src/models/similarity.py

"""
Fashion Style Similarity Calculator

역할:
- 이미지/텍스트 임베딩 간 유사도 계산
- 스타일 카테고리 분류
- 유사도 행렬 계산 (N×M)

실무 포인트:
- 코사인 유사도 = L2 정규화 후 내적
  → 벡터 크기에 무관하게 방향만 비교!
- 배치 행렬 연산으로 속도 최적화
  → 루프 없이 한 번에 계산!

자료구조:
- numpy array: 유사도 행렬 (N×M)
  → 행렬 연산 최적화
- Dict: 스타일 카테고리 매핑 (Hash Map)
  → O(1) 조회
"""

from typing import Optional
import numpy as np


# ----------------------------------------
# 패션 스타일 카테고리 정의
# 텍스트 기반 Zero-shot 분류용
# ----------------------------------------
STYLE_CATEGORIES = {
    "casual": [
        "casual style clothing",
        "everyday casual wear",
        "comfortable casual outfit",
    ],
    "formal": [
        "formal business wear",
        "professional office clothing",
        "elegant formal outfit",
    ],
    "sport": [
        "sportswear athletic clothing",
        "gym workout outfit",
        "athletic performance wear",
    ],
    "street": [
        "streetwear urban fashion",
        "hip hop street style",
        "urban casual street outfit",
    ],
    "vintage": [
        "vintage retro fashion",
        "classic vintage style clothing",
        "retro old school outfit",
    ],
}


class FashionSimilarityCalculator:
    """
    패션 스타일 유사도 계산기

    핵심 기능:
    ① 코사인 유사도 계산
    ② 유사도 행렬 계산 (N×M)
    ③ Top-K 유사 아이템 검색
    ④ 스타일 카테고리 분류

    자료구조:
    - numpy array: 유사도 행렬
      → 행렬 곱으로 한 번에 계산!
      → O(N×M×D) but 벡터화로 매우 빠름
    - Dict: 카테고리 → 텍스트 매핑
      → O(1) 조회
    """

    def __init__(self):
        print("[SimilarityCalculator] 초기화 완료!")

    def cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """
        두 벡터 간 코사인 유사도

        수식:
        similarity = (v1 · v2) / (||v1|| × ||v2||)

        L2 정규화된 벡터끼리는:
        similarity = v1 · v2  (내적만으로 충분!)

        Args:
            vec1: (D,) 벡터
            vec2: (D,) 벡터

        Returns:
            similarity: -1.0 ~ 1.0
        """
        # L2 정규화
        v1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        v2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        return float(np.dot(v1, v2))

    def cosine_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        N×M 유사도 행렬 계산

        핵심 최적화:
        행렬 곱 한 번으로 모든 쌍 계산!
        루프 없음 → O(N×M×D) 벡터 연산

        Args:
            embeddings1: (N, D) 임베딩 배열
            embeddings2: (M, D) 임베딩 배열

        Returns:
            similarity_matrix: (N, M)
            [i, j] = i번째와 j번째 유사도

        자료구조:
        numpy array (N, M)
        → 행렬 곱으로 한 번에 계산
        → 메모리 연속 할당으로 빠른 접근
        """
        # L2 정규화
        # (N, D) → 각 행을 단위 벡터로
        e1 = embeddings1 / (
            np.linalg.norm(embeddings1, axis=1, keepdims=True)
            + 1e-8
        )
        e2 = embeddings2 / (
            np.linalg.norm(embeddings2, axis=1, keepdims=True)
            + 1e-8
        )

        # 행렬 곱으로 모든 쌍의 유사도 계산
        # (N, D) × (D, M) = (N, M)
        return np.dot(e1, e2.T).astype(np.float32)

    def top_k_similar(
        self,
        query_embedding: np.ndarray,
        database_embeddings: np.ndarray,
        k: int = 10,
        exclude_idx: Optional[int] = None,
    ) -> tuple:
        """
        쿼리와 가장 유사한 Top-K 검색

        Args:
            query_embedding:     (D,) 쿼리 벡터
            database_embeddings: (N, D) DB 벡터
            k:                   반환할 결과 수
            exclude_idx:         제외할 인덱스
                                 (자기 자신 제외용)

        Returns:
            indices:     (K,) 유사한 순서 인덱스
            similarities:(K,) 유사도 점수

        자료구조:
        numpy argsort → O(N log N)
        상위 K개만 추출 → O(K)
        """
        # 쿼리와 전체 DB 유사도 계산
        # (1, D) × (D, N) = (1, N) → (N,)
        query = query_embedding.reshape(1, -1)
        similarities = self.cosine_similarity_matrix(
            query, database_embeddings
        )[0]
        # (N,) 유사도 배열

        # 자기 자신 제외
        if exclude_idx is not None:
            similarities[exclude_idx] = -2.0
            # -2.0: 절대 선택 안 되도록!

        # 내림차순 정렬 후 Top-K 추출
        # argsort: 오름차순 → [::-1]: 내림차순
        sorted_indices = np.argsort(similarities)[::-1]
        top_k_indices  = sorted_indices[:k]
        top_k_scores   = similarities[top_k_indices]

        return top_k_indices, top_k_scores

    def classify_style(
        self,
        embedding: np.ndarray,
        style_embeddings: dict,
    ) -> dict:
        """
        임베딩 → 스타일 카테고리 분류

        Zero-shot 분류!
        학습 없이 텍스트 설명만으로 분류!

        Args:
            embedding:        (D,) 이미지 임베딩
            style_embeddings: {카테고리: (D,) 임베딩}
                              encode_style_categories()
                              의 반환값

        Returns:
            result: {
                "top_style": "casual",
                "scores": {
                    "casual": 0.85,
                    "formal": 0.23,
                    ...
                }
            }
        """
        scores = {}

        for style, style_emb in style_embeddings.items():
            scores[style] = self.cosine_similarity(
                embedding, style_emb
            )

        # 가장 높은 스타일
        top_style = max(scores, key=scores.get)

        return {
            "top_style": top_style,
            "scores":    scores,
        }

    def encode_style_categories(
        self,
        encoder,
        categories: dict = STYLE_CATEGORIES,
    ) -> dict:
        """
        스타일 카테고리 텍스트 → 임베딩 변환
        서버 시작 시 한 번만 실행!

        Args:
            encoder:    FashionCLIPEncoder 인스턴스
            categories: {스타일명: [텍스트 리스트]}

        Returns:
            style_embeddings: {스타일명: (D,) 임베딩}

        자료구조:
        Dict (Hash Map): 스타일명 → 임베딩
        → O(1) 조회
        → 서버 메모리에 캐싱!
        """
        style_embeddings = {}

        for style, texts in categories.items():
            # 여러 텍스트의 평균 임베딩
            # → 더 안정적인 스타일 표현!
            text_embs = encoder.encode_texts(
                texts,
                show_progress=False,
            )
            # (텍스트 수, D) → 평균 → (D,)
            mean_emb = text_embs.mean(axis=0)

            # 평균 후 재정규화!
            mean_emb = mean_emb / (
                np.linalg.norm(mean_emb) + 1e-8
            )
            style_embeddings[style] = mean_emb

            print(f"  {style:<10}: 임베딩 완료")

        return style_embeddings

    def get_similarity_stats(
        self,
        similarity_matrix: np.ndarray,
    ) -> dict:
        """
        유사도 행렬 통계 분석
        데이터 품질 확인용

        Args:
            similarity_matrix: (N, M) 유사도 행렬

        Returns:
            stats: 통계 딕셔너리
        """
        # 대각선 제외 (자기 자신과의 유사도)
        mask = ~np.eye(
            similarity_matrix.shape[0],
            dtype=bool
        )
        off_diagonal = similarity_matrix[mask]

        return {
            "mean":   float(off_diagonal.mean()),
            "std":    float(off_diagonal.std()),
            "min":    float(off_diagonal.min()),
            "max":    float(off_diagonal.max()),
            "shape":  similarity_matrix.shape,
        }