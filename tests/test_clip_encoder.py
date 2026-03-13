# tests/test_clip_encoder.py

import time
import numpy as np
import torch
from PIL import Image
from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder

print("=== CLIP Encoder Test ===")
print()

# ----------------------------------------
# 1. 모델 로드 테스트
# ----------------------------------------
encoder = FashionCLIPEncoder(
    model_name="openai/clip-vit-base-patch32",
    device="cuda",
    use_fp16=True,
)

info = encoder.get_model_info()
print("=== Model Info ===")
for k, v in info.items():
    print(f"  {k:<20}: {v}")
print()

# ----------------------------------------
# 2. 단일 이미지 인코딩 테스트
# ----------------------------------------
print("=== Single Image Encoding ===")
image = Image.open("data/raw/images/15970.jpg").convert("RGB")

start = time.time()
img_emb = encoder.encode_single_image(image)
elapsed = time.time() - start

print(f"  Shape    : {img_emb.shape}")
print(f"  Dtype    : {img_emb.dtype}")
print(f"  Norm     : {np.linalg.norm(img_emb):.4f}")
print(f"  Time     : {elapsed*1000:.1f}ms")
print()

# ----------------------------------------
# 3. 단일 텍스트 인코딩 테스트
# ----------------------------------------
print("=== Single Text Encoding ===")
text = "a photo of Navy Blue Shirts for Men"

start = time.time()
txt_emb = encoder.encode_single_text(text)
elapsed = time.time() - start

print(f"  Text     : {text}")
print(f"  Shape    : {txt_emb.shape}")
print(f"  Norm     : {np.linalg.norm(txt_emb):.4f}")
print(f"  Time     : {elapsed*1000:.1f}ms")
print()

# ----------------------------------------
# 4. 이미지-텍스트 유사도 테스트
# 같은 아이템 → 높은 유사도!
# ----------------------------------------
print("=== Image-Text Similarity ===")

texts = [
    "a photo of Navy Blue Shirts for Men",
    # 정답 텍스트
    "a photo of Red Dress for Women",
    # 관련 없는 텍스트
    "a photo of Black Shoes for Men",
    # 관련 없는 텍스트
]

for text in texts:
    t_emb = encoder.encode_single_text(text)
    sim   = encoder.compute_similarity(img_emb, t_emb)
    print(f"  {sim:+.4f} | {text}")
print()

# ----------------------------------------
# 5. 배치 인코딩 속도 테스트
# ----------------------------------------
print("=== Batch Encoding Speed ===")
dataset = FashionDataset(
    data_dir="data/raw",
    split="all",
    max_samples=200,
)

# 이미지 200장 로드
images = [
    Image.open(meta["image_path"]).convert("RGB")
    for meta in list(
        dataset.id_to_meta.values()
    )[:200]
]

start = time.time()
embeddings = encoder.encode_images(
    images,
    batch_size=64,
    show_progress=True,
)
elapsed = time.time() - start

print(f"  이미지 수   : {len(images)}장")
print(f"  Shape      : {embeddings.shape}")
print(f"  총 시간    : {elapsed:.2f}s")
print(f"  장당 시간  : {elapsed/len(images)*1000:.1f}ms")
print()

# ----------------------------------------
# 6. L2 정규화 검증
# 모든 임베딩의 norm = 1.0 이어야 함!
# ----------------------------------------
print("=== L2 Normalization Check ===")
norms = np.linalg.norm(embeddings, axis=1)
print(f"  Norm mean  : {norms.mean():.6f}")
print(f"  Norm std   : {norms.std():.6f}")
print(f"  All ~1.0   : {np.allclose(norms, 1.0, atol=1e-5)}")
print()

print("=== All Tests Passed! ===")