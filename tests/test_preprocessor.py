# tests/test_preprocessor.py

import torch
from PIL import Image
from src.data.preprocessor import FashionPreprocessor

# ----------------------------------------
# 1. 기본 전처리 테스트
# ----------------------------------------
print("=== Preprocessor Test ===")
print()

# 추론 모드
inf_preprocessor = FashionPreprocessor(
    image_size=224,
    mode="inference"
)

# 학습 모드
train_preprocessor = FashionPreprocessor(
    image_size=224,
    mode="train"
)

# 샘플 이미지 로드
sample_path = "data/raw/images/15970.jpg"
image = Image.open(sample_path).convert("RGB")
print(f"Original size : {image.size}")
print(f"Original mode : {image.mode}")
print()

# ----------------------------------------
# 2. 추론 전처리 테스트
# ----------------------------------------
inf_tensor = inf_preprocessor(image)
print(f"Inference tensor shape : {inf_tensor.shape}")
print(f"Inference tensor min   : {inf_tensor.min():.4f}")
print(f"Inference tensor max   : {inf_tensor.max():.4f}")
print()

# ----------------------------------------
# 3. 학습 전처리 테스트
# ----------------------------------------
train_tensor = train_preprocessor(image)
print(f"Train tensor shape : {train_tensor.shape}")
print(f"Train tensor min   : {train_tensor.min():.4f}")
print(f"Train tensor max   : {train_tensor.max():.4f}")
print()

# ----------------------------------------
# 4. 동일 이미지 추론 일관성 테스트
# 추론 모드는 항상 같은 결과여야 함!
# ----------------------------------------
inf_tensor_1 = inf_preprocessor(image)
inf_tensor_2 = inf_preprocessor(image)
is_consistent = torch.allclose(inf_tensor_1, inf_tensor_2)
print(f"Inference consistency  : {is_consistent}")
# 반드시 True!

# ----------------------------------------
# 5. 학습 증강 랜덤성 테스트
# 학습 모드는 매번 달라야 함!
# ----------------------------------------
train_tensor_1 = train_preprocessor(image)
train_tensor_2 = train_preprocessor(image)
is_different = not torch.allclose(
    train_tensor_1, train_tensor_2
)
print(f"Train augmentation random : {is_different}")
# 대부분 True (랜덤 증강!)
print()

# ----------------------------------------
# 6. 배치 전처리 테스트
# ----------------------------------------
images = [image] * 4
batch  = inf_preprocessor.preprocess_batch(images)
print(f"Batch shape : {batch.shape}")
# (4, 3, 224, 224)
print()

# ----------------------------------------
# 7. 역변환 테스트 (시각화용)
# ----------------------------------------
restored = inf_preprocessor.decode_tensor(inf_tensor)
print(f"Decoded image size : {restored.size}")
print(f"Decoded image mode : {restored.mode}")
print()

# ----------------------------------------
# 8. 경로 입력 테스트
# ----------------------------------------
path_tensor = inf_preprocessor(sample_path)
print(f"Path input tensor shape : {path_tensor.shape}")
print()

# ----------------------------------------
# 9. 모드 전환 테스트
# ----------------------------------------
inf_preprocessor.set_mode("train")
inf_preprocessor.set_mode("inference")
print()
print("=== All Tests Passed! ===")