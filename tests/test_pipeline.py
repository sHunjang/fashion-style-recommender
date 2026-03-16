# tests/test_pipeline.py

"""
탐지 + 검색 통합 파이프라인 테스트
"""

import logging
from pathlib import Path
from PIL import Image

from src.detection.detector import FashionDetector
from src.detection.pipeline import FashionPipeline
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

logging.basicConfig(level=logging.INFO)
print("=== Fashion Pipeline Test ===")
print()

# ----------------------------------------
# 1. 초기화
# ----------------------------------------
print("=== 초기화 ===")

# YOLOv8 탐지기
detector = FashionDetector(
    model_path="models/yolo/fashion_detection_v2_best.pt",
    confidence=0.25,
    device="cuda",
)

# CLIP 인코더 (Fine-tuned)
encoder = FashionCLIPEncoder(
    device="cuda",
    use_fp16=True,
    lora_path="models/lora/best",
)

# FAISS 검색 엔진
indexer = FashionFAISSIndexer(
    embed_dim=512,
    nlist=50,
    nprobe=10,
)
retriever = FashionRetriever(encoder, indexer)

# 인덱스 로드
INDEX_DIR = "data/index_finetuned"
if Path(INDEX_DIR + "/fashion.index").exists():
    print("인덱스 로드 중...")
    retriever.load_index(INDEX_DIR)
else:
    print("인덱스 구축 중...")
    from src.data.dataset import FashionDataset
    dataset = FashionDataset(
        data_dir="data/raw",
        split="all",
        max_samples=500,
    )
    retriever.build_index_from_dataset(
        dataset, batch_size=32
    )
    retriever.save_index(INDEX_DIR)

# 파이프라인 생성
pipeline = FashionPipeline(
    detector=detector,
    retriever=retriever,
    top_k=3,
    min_confidence=0.25,
)
print()

# ----------------------------------------
# 2. 전신 코디 이미지로 테스트
# ----------------------------------------
print("=== 파이프라인 테스트 ===")

test_images = list(
    Path("data/fashion_detection/test/images").glob("*.jpg")
)

# 탐지 결과 있는 이미지 찾기
for img_path in test_images:
    img  = Image.open(img_path).convert("RGB")
    dets = detector.detect(img, only_searchable=True)
    if len(dets) >= 2:
        print(f"테스트 이미지: {img_path.name[:45]}")
        test_img = img
        break

# 파이프라인 실행
results, visualized = pipeline.search_and_visualize(
    test_img,
    only_searchable=True,
)

# ----------------------------------------
# 3. 결과 출력
# ----------------------------------------
print(f"\n총 탐지 아이템: {results['total_items']}개")
print(f"전체 소요 시간: {results['search_time_ms']}ms")
print()

for item in results["items"]:
    print(
        f"{'='*50}\n"
        f"아이템: {item['class_name']} "
        f"(신뢰도: {item['confidence']:.3f})\n"
        f"검색 시간: {item['search_time_ms']}ms"
    )
    print(f"유사 상품 Top-{len(item['similar'])}:")
    for r in item["similar"]:
        meta = r["metadata"]
        print(
            f"  {r['rank']}. [{r['score']:.4f}] "
            f"{meta.get('article_type', 'N/A')} | "
            f"{meta.get('name', 'N/A')[:35]}"
        )

# 시각화 저장
visualized.save("data/test_pipeline_result.jpg")
print(f"\n시각화 저장: data/test_pipeline_result.jpg")

# ----------------------------------------
# 4. 크롭 이미지 저장
# ----------------------------------------
print("\n=== 크롭 이미지 저장 ===")
for item in results["items"]:
    crop_path = (
        f"data/pipeline_crop_"
        f"{item['rank']}_{item['class_name']}.jpg"
    )
    item["crop"].save(crop_path)
    print(f"  저장: {crop_path}")

print()
print("=== Test Passed! ===")