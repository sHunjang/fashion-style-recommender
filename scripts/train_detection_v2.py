# scripts/train_detection_v2.py

"""
개선된 YOLOv8 학습
- 손상 라벨 제거
- yolov8s (small) 모델
- 클래스 가중치 적용
- epochs 증가
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def fix_dataset():
    """손상된 라벨 파일 수정"""
    broken = Path(
        "data/fashion_detection/valid/labels/"
        "1589783941431_crkvei0n719_jpg.rf."
        "ce4a385a1aa3ed2302b74a32bd409aea.txt"
    )
    if broken.exists():
        # 빈 파일로 교체 (배경 이미지로 처리)
        broken.write_text("")
        print(f"✅ 손상 라벨 수정: {broken.name}")

def main():
    fix_dataset()

    DATA_YAML   = "data/fashion_detection/data.yaml"
    MODEL       = "yolov8s.pt"   # n → s (더 정확!)
    EPOCHS      = 50
    BATCH_SIZE  = 32
    IMG_SIZE    = 640
    PROJECT_DIR = ""
    RUN_NAME    = "fashion_detection_v2"

    print("=== YOLOv8s Fashion Detection v2 ===")
    print(f"모델:   {MODEL}")
    print(f"에폭:   {EPOCHS}")
    print()

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device="cuda",
        project=PROJECT_DIR,
        name=RUN_NAME,

        # 학습률
        lr0=0.01,
        lrf=0.01,

        # 데이터 증강 강화
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,      # ← 추가! 클래스 불균형 완화
        copy_paste=0.1, # ← 추가! 소수 클래스 보강

        # 학습 안정화
        patience=30,    # 20 → 30
        save=True,
        save_period=10,
        cache=True,
        workers=0,
        amp=True,
        verbose=True,
    )

    print()
    print("=== 학습 완료! ===")
    best = (
        f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
    )
    print(f"Best mAP50: "
          f"{results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    # best.pt 복사
    import shutil
    shutil.copy(best, "models/yolo/fashion_detection_v2_best.pt")
    print("모델 저장: models/yolo/fashion_detection_v2_best.pt")


if __name__ == "__main__":
    main()