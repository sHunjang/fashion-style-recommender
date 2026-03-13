# tests/test_downloader.py

from src.data.downloader import FashionDataDownloader

downloader = FashionDataDownloader(data_dir="data/raw")

# 1. 현재 데이터셋 정보
print("=== Dataset Info ===")
downloader.get_info()
print()

# 2. 다운로드 여부 확인
print(f"Is downloaded: {downloader.is_downloaded()}")
print()

# 3. 데이터 검증 (샘플 200개만)
print("=== Validation (sample=200) ===")
report = downloader.validate(sample_size=200)
print()
print(f"Status : {report['status']}")
print(f"Valid  : {report['valid_images']}")
print(f"Missing: {report['missing_images']}")