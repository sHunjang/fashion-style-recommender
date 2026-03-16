# tests/test_puma_image.py

from src.detection.detector import FashionDetector
from PIL import Image

print("=== PUMA 이미지 탐지 테스트 ===")

detector = FashionDetector(
    model_path="models/yolo/fashion_detection_v2_best.pt",
    confidence=0.10,   # ← 매우 낮게! 0.25→0.10
    device="cuda",
)

img = Image.open(
    "data/fashion_detection/test/images/"
    "0-1-_jpg.rf.6b58dc44912edaf4b86bf10b044a3293.jpg"
).convert("RGB")

print(f"이미지 크기: {img.size}")
print()

# 전체 클래스 (only_searchable=False)
dets = detector.detect(img, only_searchable=False)
print(f"탐지 결과 (confidence=0.10): {len(dets)}개")
for d in dets:
    print(
        f"  [{d['confidence']:.3f}] "
        f"{d['class_name']} | "
        f"bbox: {d['bbox']}"
    )

# 시각화
vis = detector.visualize(img, dets)
vis.save("data/test_puma.jpg")
print("시각화: data/test_puma.jpg")