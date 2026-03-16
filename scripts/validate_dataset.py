# scripts/validate_dataset.py

"""
데이터셋 품질 검증
- 클래스별 이미지/라벨 수 확인
- 라벨 파일 정합성 확인
- 샘플 시각화
"""

import yaml
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import random

print("=== 데이터셋 검증 ===")
print()

DATA_DIR = Path("data/fashion_detection")

# ----------------------------------------
# 1. data.yaml 확인
# ----------------------------------------
print("=== data.yaml ===")
with open(DATA_DIR / "data.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

classes = data_cfg["names"]
print(f"클래스 수: {data_cfg['nc']}개")
for i, cls in enumerate(classes):
    print(f"  {i:2d}: {cls}")
print()

# ----------------------------------------
# 2. 클래스별 라벨 수 집계
# ----------------------------------------
print("=== 클래스별 라벨 수 ===")

for split in ["train", "valid", "test"]:
    label_dir = DATA_DIR / split / "labels"
    image_dir = DATA_DIR / split / "images"

    if not label_dir.exists():
        print(f"{split}: 라벨 폴더 없음!")
        continue

    label_files = list(label_dir.glob("*.txt"))
    image_files = list(image_dir.glob("*.jpg"))

    # 클래스별 카운트
    class_counter = Counter()
    empty_labels   = 0
    missing_images = 0

    for lf in label_files:
        # 대응하는 이미지 파일 확인
        img_path = image_dir / lf.with_suffix(".jpg").name
        if not img_path.exists():
            missing_images += 1

        # 라벨 파일 읽기
        content = lf.read_text().strip()
        if not content:
            empty_labels += 1
            continue

        for line in content.split("\n"):
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_counter[class_id] += 1

    print(f"\n[{split}]")
    print(f"  이미지: {len(image_files)}장")
    print(f"  라벨:   {len(label_files)}개")
    print(f"  빈 라벨: {empty_labels}개")
    print(f"  이미지 없는 라벨: {missing_images}개")
    print(f"  클래스별 라벨 수:")

    total = sum(class_counter.values())
    for class_id in sorted(class_counter.keys()):
        cls_name = classes[class_id] \
            if class_id < len(classes) else f"unknown_{class_id}"
        count = class_counter[class_id]
        bar   = "█" * (count // 10)
        print(
            f"    {class_id:2d} {cls_name:<18} "
            f"{count:4d}개  {bar}"
        )
    print(f"  총 라벨: {total}개")

# ----------------------------------------
# 3. 라벨 값 범위 검증
# ----------------------------------------
print()
print("=== 라벨 값 검증 ===")

errors = []
for split in ["train", "valid", "test"]:
    label_dir = DATA_DIR / split / "labels"
    if not label_dir.exists():
        continue

    for lf in label_dir.glob("*.txt"):
        content = lf.read_text().strip()
        if not content:
            continue
        for i, line in enumerate(content.split("\n")):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(
                    f"{lf.name} line {i+1}: "
                    f"컬럼 수 오류 ({len(parts)}개)"
                )
                continue
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            # 클래스 ID 범위 확인
            if class_id >= data_cfg["nc"]:
                errors.append(
                    f"{lf.name}: "
                    f"클래스 ID 오류 ({class_id})"
                )

            # 좌표 범위 확인 (0~1)
            for val in [x, y, w, h]:
                if not (0 <= val <= 1):
                    errors.append(
                        f"{lf.name}: "
                        f"좌표 범위 오류 ({val})"
                    )

if errors:
    print(f"오류 {len(errors)}개 발견!")
    for e in errors[:10]:
        print(f"  ❌ {e}")
else:
    print("✅ 라벨 값 모두 정상!")

# ----------------------------------------
# 4. 샘플 시각화 저장
# ----------------------------------------
print()
print("=== 샘플 시각화 ===")

sample_dir = Path("data/dataset_samples")
sample_dir.mkdir(exist_ok=True)

train_images = list(
    (DATA_DIR / "train" / "images").glob("*.jpg")
)
samples = random.sample(
    train_images, min(9, len(train_images))
)

for img_path in samples:
    label_path = (
        DATA_DIR / "train" / "labels" /
        img_path.with_suffix(".txt").name
    )

    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if label_path.exists():
        content = label_path.read_text().strip()
        for line in content.split("\n"):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id    = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # YOLO → 픽셀 좌표 변환
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            cls_name = classes[class_id] \
                if class_id < len(classes) \
                else f"cls_{class_id}"

            draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
            draw.text((x1, y1), cls_name, fill="red")

    save_path = sample_dir / img_path.name
    img.save(save_path)

print(f"샘플 저장: data/dataset_samples/ ({len(samples)}장)")
print("→ 이미지 열어서 라벨이 올바른지 확인하세요!")
print()
print("=== 검증 완료! ===")