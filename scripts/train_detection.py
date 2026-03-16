# scripts/train_detection.py

"""
YOLOv8 패션 탐지 모델 파인튜닝

실행:
python scripts/train_detection.py
"""

import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

# ----------------------------------------
# 설정
# ----------------------------------------
DATA_YAML   = "data/fashion_detection/data.yaml"
MODEL       = "yolov8n.pt"    # nano: 빠름, 경량
EPOCHS      = 50
BATCH_SIZE  = 16
IMG_SIZE    = 640
PROJECT_DIR = "models/yolo"
RUN_NAME    = "fashion_detection_v1"

print("=== YOLOv8 Fashion Detection Fine-tuning ===")
print(f"모델:      {MODEL}")
print(f"에폭:      {EPOCHS}")
print(f"배치:      {BATCH_SIZE}")
print(f"이미지:    {IMG_SIZE}×{IMG_SIZE}")
print()

# ----------------------------------------
# 모델 로드
# ----------------------------------------
model = YOLO(MODEL)

# ----------------------------------------
# 파인튜닝 실행
# ----------------------------------------
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device="cuda",
    project=PROJECT_DIR,
    name=RUN_NAME,

    # 학습률 설정
    lr0=0.01,       # 초기 학습률
    lrf=0.01,       # 최종 학습률 (lr0 * lrf)

    # 데이터 증강
    hsv_h=0.015,    # 색조 변환
    hsv_s=0.7,      # 채도 변환
    hsv_v=0.4,      # 명도 변환
    flipud=0.0,     # 상하 반전 (패션은 안 씀!)
    fliplr=0.5,     # 좌우 반전

    # 학습 안정화
    patience=20,    # Early stopping
    save=True,
    save_period=10, # 10 에폭마다 저장

    # 성능 최적화
    cache=True,     # 이미지 캐싱 (빠른 학습!)
    workers=0,
    amp=True,       # fp16 자동 혼합 정밀도

    verbose=True,
)

print()
print("=== 학습 완료! ===")
print(f"저장 경로: {PROJECT_DIR}/{RUN_NAME}")
print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")