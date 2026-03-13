# src/data/dataset.py

"""
Fashion Product Dataset

데이터셋 구조:
- 44,424개 패션 이미지
- 10개 메타데이터 컬럼
  id, gender, masterCategory, subCategory,
  articleType, baseColour, season, year,
  usage, productDisplayName

자료구조:
- DataFrame: 메타데이터 테이블 관리
- Hash Map (dict): id → 메타데이터 O(1) 조회
- List: 이미지 경로 순서 관리
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ----------------------------------------
# 카테고리 필터
# 실습에서 사용할 카테고리만 선택
# ----------------------------------------
VALID_CATEGORIES = ["Apparel", "Footwear", "Accessories"]

# 스타일 텍스트 템플릿
# CLIP 텍스트 인코더 입력용
# "a photo of {색상} {articleType}" 형식
TEXT_TEMPLATE = "a photo of {colour} {article} for {gender}"


class FashionDataset(Dataset):
    """
    Fashion Product Image Dataset

    역할:
    - 이미지 + 메타데이터 로드
    - 전처리 파이프라인 적용
    - CLIP 입력용 텍스트 설명 생성

    자료구조:
    ① DataFrame: 전체 메타데이터 테이블
    ② dict (Hash Map): id → row 매핑 O(1) 조회
    ③ list: 유효한 이미지 경로 목록
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
        categories: list = VALID_CATEGORIES,
    ):
        """
        Args:
            data_dir:    데이터 루트 경로 (data/raw)
            split:       'train' | 'val' | 'all'
            max_samples: 최대 샘플 수 (None=전체)
            image_size:  CLIP 입력 이미지 크기
            categories:  사용할 masterCategory 목록
        """
        self.data_dir   = Path(data_dir)
        self.image_dir  = self.data_dir / "images"
        self.image_size = image_size

        # ----------------------------------------
        # 1. 메타데이터 로드
        # ----------------------------------------
        df = pd.read_csv(
            self.data_dir / "styles.csv",
            on_bad_lines="skip"
        )

        # ----------------------------------------
        # 2. 카테고리 필터링
        # Apparel, Footwear, Accessories만 사용
        # ----------------------------------------
        df = df[df["masterCategory"].isin(categories)]
        df = df.reset_index(drop=True)

        # ----------------------------------------
        # 3. 실제 이미지 파일 존재 여부 확인
        # 자료구조: list comprehension
        # ----------------------------------------
        valid_mask = df["id"].apply(
            lambda x: (self.image_dir / f"{x}.jpg").exists()
        )
        df = df[valid_mask].reset_index(drop=True)

        # ----------------------------------------
        # 4. 결측값 처리
        # ----------------------------------------
        df["baseColour"]  = df["baseColour"].fillna("Unknown")
        df["articleType"] = df["articleType"].fillna("item")
        df["gender"]      = df["gender"].fillna("Unisex")
        df["usage"]       = df["usage"].fillna("Casual")
        df["season"]      = df["season"].fillna("All Season")
        df["productDisplayName"] = \
            df["productDisplayName"].fillna("")

        # ----------------------------------------
        # 5. train / val 분할
        # 80% train, 20% val
        # ----------------------------------------
        n_total = len(df)
        n_train = int(n_total * 0.8)

        if split == "train":
            df = df.iloc[:n_train]
        elif split == "val":
            df = df.iloc[n_train:]
        # split == "all": 전체 사용

        # ----------------------------------------
        # 6. max_samples 제한
        # ----------------------------------------
        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.df = df.reset_index(drop=True)

        # ----------------------------------------
        # 자료구조: Hash Map (dict)
        # id → 메타데이터 행 O(1) 조회
        # 검색 결과에서 메타데이터 즉시 반환용
        # ----------------------------------------
        self.id_to_meta = {
            int(row["id"]): {
                "id":           int(row["id"]),
                "image_path":   str(
                    self.image_dir / f"{int(row['id'])}.jpg"
                ),
                "gender":       row["gender"],
                "category":     row["masterCategory"],
                "sub_category": row["subCategory"],
                "article_type": row["articleType"],
                "colour":       row["baseColour"],
                "season":       row["season"],
                "usage":        row["usage"],
                "name":         row["productDisplayName"],
                "text":         self._make_text(row),
            }
            for _, row in self.df.iterrows()
        }

        # ----------------------------------------
        # 자료구조: list
        # 순서 있는 이미지 ID 목록
        # DataLoader 인덱싱용
        # ----------------------------------------
        self.image_ids = self.df["id"].tolist()

        # ----------------------------------------
        # 이미지 전처리 파이프라인
        # CLIP 표준 전처리
        # ----------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275,  0.40821073),
                std= (0.26862954, 0.26130258, 0.27577711)
                # CLIP 공식 정규화 값!
                # ImageNet 값과 다름 주의!
            ),
        ])

        print(f"[FashionDataset] {split} 세트 로드 완료!")
        print(f"  전체 샘플 수: {len(self.df):,}개")
        print(f"  카테고리: {self.df['masterCategory'].value_counts().to_dict()}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: 데이터셋 인덱스

        Returns:
            dict:
                image:      (3, 224, 224) Tensor
                text:       텍스트 설명 문자열
                image_id:   이미지 ID (int)
                metadata:   메타데이터 dict
        """
        image_id = int(self.image_ids[idx])
        meta     = self.id_to_meta[image_id]

        # 이미지 로드
        image = Image.open(
            meta["image_path"]
        ).convert("RGB")

        # 전처리
        image_tensor = self.transform(image)

        return {
            "image":    image_tensor,
            "text":     meta["text"],
            "image_id": image_id,
            "metadata": meta,
        }

    def get_metadata(self, image_id: int) -> dict:
        """
        이미지 ID로 메타데이터 조회
        자료구조: Hash Map O(1) 조회

        Args:
            image_id: 이미지 ID

        Returns:
            메타데이터 dict
        """
        return self.id_to_meta.get(image_id, {})

    def get_all_metadata(self) -> list:
        """
        전체 메타데이터 리스트 반환
        FAISS 인덱스 구축 시 사용

        Returns:
            list of dict
        """
        return [
            self.id_to_meta[int(img_id)]
            for img_id in self.image_ids
        ]

    def _make_text(self, row: pd.Series) -> str:
        """
        메타데이터 → CLIP 텍스트 설명 생성

        Args:
            row: DataFrame 행

        Returns:
            텍스트 설명
            예: "a photo of Navy Blue Shirts for Men"
        """
        return TEXT_TEMPLATE.format(
            colour=row["baseColour"],
            article=row["articleType"],
            gender=row["gender"],
        )

    def get_stats(self) -> dict:
        """데이터셋 통계 반환"""
        return {
            "total":      len(self.df),
            "categories": self.df["masterCategory"]
                              .value_counts().to_dict(),
            "genders":    self.df["gender"]
                              .value_counts().to_dict(),
            "top_articles": self.df["articleType"]
                                .value_counts()
                                .head(10).to_dict(),
            "colours":    self.df["baseColour"]
                              .value_counts()
                              .head(10).to_dict(),
        }