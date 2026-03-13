# tests/test_dataset.py

from src.data.dataset import FashionDataset

# 데이터셋 로드 테스트
dataset = FashionDataset(
    data_dir="data/raw",
    split="all",
    max_samples=100,
)

# 샘플 확인
sample = dataset[0]
print()
print("=== Sample Check ===")
print(f"Image shape : {sample['image'].shape}")
print(f"Text        : {sample['text']}")
print(f"Image ID    : {sample['image_id']}")
print(f"Category    : {sample['metadata']['category']}")
print(f"Colour      : {sample['metadata']['colour']}")
print()

# Hash Map 조회 테스트 O(1)
image_id = sample["image_id"]
meta = dataset.get_metadata(image_id)
print("=== Hash Map Lookup ===")
print(f"ID     : {meta['id']}")
print(f"Name   : {meta['name']}")
print(f"Usage  : {meta['usage']}")
print(f"Season : {meta['season']}")
print()

# 통계 확인
stats = dataset.get_stats()
print("=== Dataset Stats ===")
for k, v in stats.items():
    print(f"{k}: {v}")