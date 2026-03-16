# tests/test_detector_finetuned.py

from src.detection.detector import FashionDetector
from pathlib import Path
from PIL import Image

print("=== 파인튜닝 모델 탐지 테스트 ===")

detector = FashionDetector(
    model_path="models/yolo/fashion_detection_v2_best.pt",
    confidence=0.25,
    device="cuda",
)

test_images = list(
    Path("data/fashion_detection/test/images").glob("*.jpg")
)
print(f"테스트 이미지 수: {len(test_images)}")
print()

for img_path in test_images[:5]:
    img  = Image.open(img_path).convert("RGB")
    dets = detector.detect(img, only_searchable=True)
    print(f"{img_path.name[:45]}")
    if dets:
        for d in dets:
            print(
                f"  [{d['confidence']:.3f}] "
                f"{d['class_name']}"
            )
    else:
        print("  탐지 없음")
    print()

# 시각화 저장
img  = Image.open(test_images[0]).convert("RGB")
dets = detector.detect(img, only_searchable=False)
vis  = detector.visualize(img, dets)
vis.save("data/test_finetuned_detection.jpg")
print("시각화 저장: data/test_finetuned_detection.jpg")
print()
print("=== Test Passed! ===")