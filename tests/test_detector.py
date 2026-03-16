# tests/test_detector.py

import logging
from pathlib import Path
from PIL import Image
from src.detection.detector import FashionDetector

logging.basicConfig(level=logging.INFO)
print("=== Fashion Detector Test ===")
print()

# ----------------------------------------
# 1. 기본 모델 테스트
# ----------------------------------------
print("=== 기본 모델 테스트 ===")
detector = FashionDetector(
    model_path="yolov8n.pt",
    confidence=0.25,
    device="cuda",
)

# 패션 탐지 데이터셋 이미지 사용!
test_images = list(
    Path("data/fashion_detection/test/images").glob("*.jpg")
)
print(f"테스트 이미지 수: {len(test_images)}")

# 첫 번째 이미지로 테스트
test_img = Image.open(test_images[0]).convert("RGB")
print(f"이미지 경로: {test_images[0].name}")
print(f"이미지 크기: {test_img.size}")
print()

# ----------------------------------------
# 2. COCO 모델로 탐지 (person 등 감지)
# ----------------------------------------
print("=== COCO 모델 탐지 (파인튜닝 전) ===")
detections = detector.detect(
    test_img,
    only_searchable=False,
)
print(f"탐지 결과: {len(detections)}개")
for det in detections:
    print(
        f"  {det['rank']}. [{det['confidence']:.3f}] "
        f"{det['class_name']} | "
        f"bbox: {det['bbox']}"
    )
print()

# ----------------------------------------
# 3. 시각화 저장
# ----------------------------------------
print("=== 시각화 테스트 ===")
visualized = detector.visualize(
    test_img,
    detections,
)
visualized.save("data/test_detection_coco.jpg")
print(f"시각화 저장: data/test_detection_coco.jpg")
print()

# ----------------------------------------
# 4. 여러 이미지 테스트
# ----------------------------------------
print("=== 5장 배치 테스트 ===")
for i, img_path in enumerate(test_images[:5], 1):
    img = Image.open(img_path).convert("RGB")
    dets = detector.detect(img, only_searchable=False)
    print(
        f"  {i}. {img_path.name[:40]} | "
        f"크기: {img.size} | "
        f"탐지: {len(dets)}개"
    )
print()

print("=== Test Passed! ===")