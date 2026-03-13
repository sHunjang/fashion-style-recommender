# src/data/downloader.py

"""
Fashion Dataset Downloader & Validator

역할:
- 데이터셋 다운로드 자동화 (Kaggle API)
- 다운로드된 데이터 무결성 검증
- 데이터셋 통계 리포트 생성

실무 포인트:
- 이미 다운로드된 경우 스킵 (중복 방지)
- 깨진 이미지 자동 감지 및 리포트
- 팀원 누구나 동일한 환경 재현 가능!

자료구조:
- Dict: 검증 결과 집계 (Hash Map)
- List: 깨진 파일 목록 관리
- Set: 중복 ID 감지 O(1)
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm


class FashionDataDownloader:
    """
    Kaggle 패션 데이터셋 다운로더 및 검증기

    지원 데이터셋:
    - fashion-product-images-small (44,424장)
    - fashion-product-images (대용량 전체)
    """

    # Kaggle 데이터셋 정보
    DATASETS = {
        "small": {
            "name": "paramaggarwal/fashion-product-images-small",
            "zip":  "fashion-product-images-small.zip",
            "size": "565MB",
            "count": 44424,
        },
        "full": {
            "name": "paramaggarwal/fashion-product-images",
            "zip":  "fashion-product-images.zip",
            "size": "25GB",
            "count": 44441,
        },
    }

    def __init__(self, data_dir: str = "data/raw"):
        """
        Args:
            data_dir: 데이터 저장 경로
        """
        self.data_dir  = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.csv_path  = self.data_dir / "styles.csv"

        # 데이터 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def is_downloaded(self) -> bool:
        """
        데이터셋이 이미 다운로드됐는지 확인

        Returns:
            True: 이미 존재
            False: 다운로드 필요
        """
        return (
            self.image_dir.exists()
            and self.csv_path.exists()
            and len(list(self.image_dir.glob("*.jpg"))) > 0
        )

    def download(
        self,
        dataset: str = "small",
        force: bool = False,
    ) -> None:
        """
        Kaggle API로 데이터셋 다운로드

        Args:
            dataset: 'small' | 'full'
            force:   True면 이미 있어도 재다운로드

        사전 조건:
        kaggle.json이 ~/.kaggle/ 에 있어야 함!
        https://www.kaggle.com/settings → API
        """
        # 이미 다운로드된 경우 스킵
        if self.is_downloaded() and not force:
            print("[Downloader] 데이터셋 이미 존재!")
            print(f"  경로: {self.data_dir}")
            print("  재다운로드: force=True 옵션 사용")
            return

        info = self.DATASETS[dataset]
        print(f"[Downloader] 다운로드 시작!")
        print(f"  데이터셋: {info['name']}")
        print(f"  용량:     {info['size']}")
        print(f"  이미지 수: {info['count']:,}장")
        print()

        # Kaggle API 사용 가능 여부 확인
        try:
            import kaggle
        except ImportError:
            print("[ERROR] kaggle 패키지 없음!")
            print("  pip install kaggle")
            return
        except Exception as e:
            print(f"[ERROR] Kaggle API 오류: {e}")
            print("  ~/.kaggle/kaggle.json 확인!")
            return

        # 다운로드
        zip_path = self.data_dir / info["zip"]
        os.system(
            f"kaggle datasets download "
            f"-d {info['name']} "
            f"-p {self.data_dir}"
        )

        # 압축 해제
        if zip_path.exists():
            print(f"\n[Downloader] 압축 해제 중...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            zip_path.unlink()
            # 압축 파일 삭제 (용량 확보)
            print("[Downloader] 압축 해제 완료!")
        else:
            print("[ERROR] zip 파일 없음! 다운로드 실패")

    def validate(
        self,
        sample_size: Optional[int] = None,
    ) -> dict:
        """
        다운로드된 데이터셋 무결성 검증

        검증 항목:
        ① CSV 파일 존재 및 컬럼 확인
        ② 이미지 파일 존재 여부
        ③ 깨진 이미지 감지
        ④ 중복 ID 감지
        ⑤ 결측값 확인

        Args:
            sample_size: 검증할 샘플 수
                         None이면 전체 검증

        Returns:
            report: 검증 결과 딕셔너리

        자료구조:
        - Dict: 검증 결과 집계 (Hash Map)
        - List: 깨진 파일 목록
        - Set: 중복 ID 감지 O(1)
        """
        print("[Downloader] 데이터셋 검증 시작...")
        print()

        # 자료구조: Dict (검증 결과 집계)
        report = {
            "total_csv_rows":    0,
            "valid_images":      0,
            "missing_images":    0,
            "corrupted_images":  0,
            "duplicate_ids":     0,
            "missing_values":    {},
            "corrupted_list":    [],  # List
            "missing_list":      [],  # List
            "category_dist":     {},
            "status":            "unknown",
        }

        # ----------------------------------------
        # ① CSV 확인
        # ----------------------------------------
        if not self.csv_path.exists():
            print("[ERROR] styles.csv 없음!")
            report["status"] = "failed"
            return report

        df = pd.read_csv(
            self.csv_path,
            on_bad_lines="skip"
        )
        report["total_csv_rows"] = len(df)
        print(f"CSV 행 수: {len(df):,}개")

        # ----------------------------------------
        # ② 중복 ID 감지
        # 자료구조: Set → O(1) 중복 확인
        # ----------------------------------------
        id_set = set()
        duplicate_count = 0
        for img_id in df["id"]:
            if img_id in id_set:
                duplicate_count += 1
            id_set.add(img_id)
        report["duplicate_ids"] = duplicate_count
        print(f"중복 ID: {duplicate_count}개")

        # ----------------------------------------
        # ③ 결측값 확인
        # ----------------------------------------
        missing = df.isnull().sum()
        report["missing_values"] = {
            col: int(cnt)
            for col, cnt in missing.items()
            if cnt > 0
        }
        print(f"결측값: {report['missing_values']}")

        # ----------------------------------------
        # ④ 카테고리 분포
        # ----------------------------------------
        report["category_dist"] = (
            df["masterCategory"]
            .value_counts()
            .to_dict()
        )

        # ----------------------------------------
        # ⑤ 이미지 파일 검증
        # 자료구조: List → 깨진 파일 목록 수집
        # ----------------------------------------
        print()
        print("이미지 파일 검증 중...")

        # 샘플링
        if sample_size:
            df_check = df.sample(
                min(sample_size, len(df)),
                random_state=42
            )
        else:
            df_check = df

        valid      = 0
        missing    = 0
        corrupted  = 0

        for _, row in tqdm(
            df_check.iterrows(),
            total=len(df_check),
            desc="검증"
        ):
            img_path = self.image_dir / f"{row['id']}.jpg"

            # 파일 존재 확인
            if not img_path.exists():
                missing += 1
                report["missing_list"].append(
                    str(img_path)
                )
                continue

            # 이미지 열기 시도 (깨진 파일 감지)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    # verify(): 파일 무결성 확인
                valid += 1
            except Exception:
                corrupted += 1
                report["corrupted_list"].append(
                    str(img_path)
                )

        report["valid_images"]     = valid
        report["missing_images"]   = missing
        report["corrupted_images"] = corrupted

        # ----------------------------------------
        # 최종 상태 판정
        # ----------------------------------------
        total_checked = valid + missing + corrupted
        valid_ratio   = valid / total_checked \
                        if total_checked > 0 else 0

        if valid_ratio >= 0.95:
            report["status"] = "good"
        elif valid_ratio >= 0.80:
            report["status"] = "warning"
        else:
            report["status"] = "failed"

        # ----------------------------------------
        # 리포트 출력
        # ----------------------------------------
        print()
        print("=" * 45)
        print("검증 리포트")
        print("=" * 45)
        print(f"전체 CSV 행:     {report['total_csv_rows']:,}")
        print(f"검증한 이미지:   {total_checked:,}")
        print(f"정상 이미지:     {valid:,} "
              f"({valid_ratio*100:.1f}%)")
        print(f"누락 이미지:     {missing:,}")
        print(f"손상 이미지:     {corrupted:,}")
        print(f"중복 ID:         {duplicate_count:,}")
        print(f"상태:            {report['status'].upper()}")
        print("=" * 45)
        print("카테고리 분포:")
        for cat, cnt in report["category_dist"].items():
            print(f"  {cat:<20}: {cnt:,}")
        print("=" * 45)

        return report

    def get_info(self) -> None:
        """현재 데이터셋 정보 출력"""
        if not self.is_downloaded():
            print("[Downloader] 데이터셋 없음!")
            print("  download() 메서드로 다운로드하세요.")
            return

        images = list(self.image_dir.glob("*.jpg"))
        size_mb = sum(
            f.stat().st_size
            for f in images
        ) / (1024 * 1024)

        print("=" * 45)
        print("데이터셋 정보")
        print("=" * 45)
        print(f"경로:        {self.data_dir}")
        print(f"이미지 수:   {len(images):,}장")
        print(f"용량:        {size_mb:.1f} MB")
        print("=" * 45)